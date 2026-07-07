#include "ggft_solver.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace gft {
namespace {

double dot(const std::vector<double>& lhs, const std::vector<double>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("dot(): vector sizes do not match");
    }

    double result = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        result += lhs[i] * rhs[i];
    }
    return result;
}

void validate_finite(const std::vector<double>& values, const char* name) {
    for (double value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(std::string(name) + " contains non-finite values");
        }
    }
}

}  // namespace

HomogeneousCubeGFTSolver::HomogeneousCubeGFTSolver(int resolution)
    : resolution_(resolution),
      unknown_count_(static_cast<std::size_t>(resolution) *
                     static_cast<std::size_t>(resolution) *
                     static_cast<std::size_t>(resolution)) {
    if (resolution_ < 3) {
        throw std::invalid_argument("resolution must be at least 3");
    }
}

int HomogeneousCubeGFTSolver::resolution() const noexcept {
    return resolution_;
}

std::size_t HomogeneousCubeGFTSolver::index(int i, int j, int k) const noexcept {
    const auto n = static_cast<std::size_t>(resolution_);
    return static_cast<std::size_t>(i) +
           n * (static_cast<std::size_t>(j) + n * static_cast<std::size_t>(k));
}

void HomogeneousCubeGFTSolver::apply_laplacian(
    const std::vector<double>& input,
    std::vector<double>& output
) const {
    if (input.size() != unknown_count_) {
        throw std::invalid_argument("apply_laplacian(): invalid input size");
    }

    output.assign(unknown_count_, 0.0);

    for (int k = 0; k < resolution_; ++k) {
        for (int j = 0; j < resolution_; ++j) {
            for (int i = 0; i < resolution_; ++i) {
                const std::size_t row = index(i, j, k);
                double value = 6.0 * input[row];

                if (i > 0) {
                    value -= input[index(i - 1, j, k)];
                }
                if (i + 1 < resolution_) {
                    value -= input[index(i + 1, j, k)];
                }
                if (j > 0) {
                    value -= input[index(i, j - 1, k)];
                }
                if (j + 1 < resolution_) {
                    value -= input[index(i, j + 1, k)];
                }
                if (k > 0) {
                    value -= input[index(i, j, k - 1)];
                }
                if (k + 1 < resolution_) {
                    value -= input[index(i, j, k + 1)];
                }

                output[row] = value;
            }
        }
    }
}

std::vector<double> HomogeneousCubeGFTSolver::build_centre_source() const {
    std::vector<double> source(unknown_count_, 0.0);

    if (resolution_ % 2 == 1) {
        const int centre = resolution_ / 2;
        source[index(centre, centre, centre)] = 1.0;
        return source;
    }

    const int lower = resolution_ / 2 - 1;
    const int upper = resolution_ / 2;
    for (int k : {lower, upper}) {
        for (int j : {lower, upper}) {
            for (int i : {lower, upper}) {
                source[index(i, j, k)] = 1.0 / 8.0;
            }
        }
    }

    return source;
}

std::vector<double> HomogeneousCubeGFTSolver::extract_boundary_probabilities(
    const std::vector<double>& green_values
) const {
    if (green_values.size() != unknown_count_) {
        throw std::invalid_argument("extract_boundary_probabilities(): invalid input size");
    }

    const std::size_t panels_per_face =
        static_cast<std::size_t>(resolution_) * static_cast<std::size_t>(resolution_);
    std::vector<double> probabilities;
    probabilities.reserve(6 * panels_per_face);

    // z-
    for (int j = 0; j < resolution_; ++j) {
        for (int i = 0; i < resolution_; ++i) {
            probabilities.push_back(green_values[index(i, j, 0)]);
        }
    }
    // z+
    for (int j = 0; j < resolution_; ++j) {
        for (int i = 0; i < resolution_; ++i) {
            probabilities.push_back(green_values[index(i, j, resolution_ - 1)]);
        }
    }
    // y-
    for (int k = 0; k < resolution_; ++k) {
        for (int i = 0; i < resolution_; ++i) {
            probabilities.push_back(green_values[index(i, 0, k)]);
        }
    }
    // y+
    for (int k = 0; k < resolution_; ++k) {
        for (int i = 0; i < resolution_; ++i) {
            probabilities.push_back(green_values[index(i, resolution_ - 1, k)]);
        }
    }
    // x-
    for (int k = 0; k < resolution_; ++k) {
        for (int j = 0; j < resolution_; ++j) {
            probabilities.push_back(green_values[index(0, j, k)]);
        }
    }
    // x+
    for (int k = 0; k < resolution_; ++k) {
        for (int j = 0; j < resolution_; ++j) {
            probabilities.push_back(green_values[index(resolution_ - 1, j, k)]);
        }
    }

    validate_finite(probabilities, "boundary probabilities");

    constexpr double negative_tolerance = 1.0e-12;
    for (double& probability : probabilities) {
        if (probability < -negative_tolerance) {
            throw std::runtime_error("solver produced a negative boundary probability");
        }
        probability = std::max(0.0, probability);
    }

    const double sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
    if (!std::isfinite(sum) || sum <= 0.0) {
        throw std::runtime_error("boundary probability sum is not positive");
    }

    for (double& probability : probabilities) {
        probability /= sum;
    }

    return probabilities;
}

std::vector<double> HomogeneousCubeGFTSolver::solve(
    double relative_tolerance,
    int max_iterations
) const {
    if (!(relative_tolerance > 0.0) || !std::isfinite(relative_tolerance)) {
        throw std::invalid_argument("relative_tolerance must be positive and finite");
    }
    if (max_iterations <= 0) {
        throw std::invalid_argument("max_iterations must be positive");
    }

    const std::vector<double> source = build_centre_source();
    std::vector<double> solution(unknown_count_, 0.0);
    std::vector<double> residual = source;
    std::vector<double> direction = residual;
    std::vector<double> matrix_direction(unknown_count_, 0.0);

    double residual_squared = dot(residual, residual);
    const double source_norm = std::sqrt(residual_squared);
    const double target_norm = relative_tolerance * std::max(1.0, source_norm);

    bool converged = std::sqrt(residual_squared) <= target_norm;

    for (int iteration = 0; iteration < max_iterations && !converged; ++iteration) {
        apply_laplacian(direction, matrix_direction);
        const double denominator = dot(direction, matrix_direction);

        if (!std::isfinite(denominator) || denominator <= 0.0) {
            throw std::runtime_error("conjugate-gradient solver encountered a non-positive denominator");
        }

        const double alpha = residual_squared / denominator;
        for (std::size_t i = 0; i < unknown_count_; ++i) {
            solution[i] += alpha * direction[i];
            residual[i] -= alpha * matrix_direction[i];
        }

        const double next_residual_squared = dot(residual, residual);
        if (!std::isfinite(next_residual_squared)) {
            throw std::runtime_error("conjugate-gradient residual became non-finite");
        }

        converged = std::sqrt(next_residual_squared) <= target_norm;
        if (!converged) {
            const double beta = next_residual_squared / residual_squared;
            for (std::size_t i = 0; i < unknown_count_; ++i) {
                direction[i] = residual[i] + beta * direction[i];
            }
        }

        residual_squared = next_residual_squared;
    }

    if (!converged) {
        throw std::runtime_error("conjugate-gradient solver did not converge");
    }

    validate_finite(solution, "Green-function solution");
    return extract_boundary_probabilities(solution);
}

void HomogeneousCubeGFTSolver::write_deeprwcap_binary(
    const std::string& output_path,
    const std::vector<double>& probabilities
) const {
    const std::size_t expected_probability_count =
        6ULL * static_cast<std::size_t>(resolution_) * static_cast<std::size_t>(resolution_);
    if (probabilities.size() != expected_probability_count) {
        throw std::invalid_argument("invalid number of boundary probabilities");
    }

    const std::filesystem::path path(output_path);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("could not open output file: " + output_path);
    }

    const double n_header = static_cast<double>(resolution_);
    const double block_width_header = 1.0;
    output.write(reinterpret_cast<const char*>(&n_header), sizeof(double));
    output.write(reinterpret_cast<const char*>(&block_width_header), sizeof(double));

    // Preserve the original DeepRWCap dataset layout even though this solver
    // contains no dielectric interfaces: every dielectric cell is epsilon=1.
    const double homogeneous_dielectric = 1.0;
    for (std::size_t i = 0; i < unknown_count_; ++i) {
        output.write(
            reinterpret_cast<const char*>(&homogeneous_dielectric),
            sizeof(double)
        );
    }

    output.write(
        reinterpret_cast<const char*>(probabilities.data()),
        static_cast<std::streamsize>(probabilities.size() * sizeof(double))
    );

    if (!output) {
        throw std::runtime_error("failed while writing output file: " + output_path);
    }
}

}  // namespace gft
