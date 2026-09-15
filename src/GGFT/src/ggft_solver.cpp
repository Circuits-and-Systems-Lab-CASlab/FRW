#include "ggft_solver.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace gft {
namespace {

constexpr std::array<char, 8> kMagic = {'G', 'G', 'F', 'T', 'T', 'B', 'L', '1'};
constexpr std::uint32_t kFormatVersion = 1;

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

double harmonic_mean(double lhs, double rhs) {
    const double sum = lhs + rhs;
    if (!(lhs > 0.0) || !(rhs > 0.0) || !std::isfinite(sum)) {
        throw std::runtime_error("dielectric values must be positive and finite");
    }
    return 2.0 * lhs * rhs / sum;
}

template <typename T>
void write_scalar(std::ofstream& output, const T& value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void write_vector(std::ofstream& output, const std::vector<double>& values) {
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(double))
    );
}

}  // namespace

GGFTSolver::GGFTSolver(
    int resolution,
    int block_width,
    std::vector<double> dielectric
)
    : resolution_(resolution),
      block_width_(block_width),
      dielectric_resolution_(
          block_width > 0 && resolution > 0 ? resolution / block_width : 0
      ),
      unknown_count_(static_cast<std::size_t>(resolution) *
                     static_cast<std::size_t>(resolution) *
                     static_cast<std::size_t>(resolution)),
      dielectric_(std::move(dielectric)) {
    if (resolution_ < 3) {
        throw std::invalid_argument("resolution must be at least 3");
    }
    if (block_width_ <= 0 || resolution_ % block_width_ != 0) {
        throw std::invalid_argument(
            "block_width must be positive and divide the resolution"
        );
    }

    const std::size_t expected_dielectric_count =
        static_cast<std::size_t>(dielectric_resolution_) *
        static_cast<std::size_t>(dielectric_resolution_) *
        static_cast<std::size_t>(dielectric_resolution_);

    if (dielectric_.empty()) {
        dielectric_.assign(expected_dielectric_count, 1.0);
    }
    if (dielectric_.size() != expected_dielectric_count) {
        throw std::invalid_argument(
            "dielectric array size does not match (resolution / block_width)^3"
        );
    }

    for (double value : dielectric_) {
        if (!(value > 0.0) || !std::isfinite(value)) {
            throw std::invalid_argument(
                "all dielectric values must be positive and finite"
            );
        }
    }
}

int GGFTSolver::resolution() const {
    return resolution_;
}

int GGFTSolver::block_width() const {
    return block_width_;
}

int GGFTSolver::dielectric_resolution() const {
    return dielectric_resolution_;
}

const std::vector<double>& GGFTSolver::dielectric() const {
    return dielectric_;
}

std::size_t GGFTSolver::index(int i, int j, int k) const {
    const auto n = static_cast<std::size_t>(resolution_);
    return static_cast<std::size_t>(i) +
           n * (static_cast<std::size_t>(j) + n * static_cast<std::size_t>(k));
}

std::size_t GGFTSolver::dielectric_index(int i, int j, int k) const {
    const auto n = static_cast<std::size_t>(dielectric_resolution_);
    const auto bi = static_cast<std::size_t>(i / block_width_);
    const auto bj = static_cast<std::size_t>(j / block_width_);
    const auto bk = static_cast<std::size_t>(k / block_width_);
    return bi + n * (bj + n * bk);
}

double GGFTSolver::dielectric_at(int i, int j, int k) const {
    return dielectric_[dielectric_index(i, j, k)];
}

void GGFTSolver::apply_operator(
    const std::vector<double>& input,
    std::vector<double>& output
) const {
    if (input.size() != unknown_count_) {
        throw std::invalid_argument("apply_operator(): invalid input size");
    }

    output.assign(unknown_count_, 0.0);

    for (int k = 0; k < resolution_; ++k) {
        for (int j = 0; j < resolution_; ++j) {
            for (int i = 0; i < resolution_; ++i) {
                const std::size_t row = index(i, j, k);
                const double epsilon_self = dielectric_at(i, j, k);
                double diagonal = 0.0;
                double value = 0.0;

                const auto add_neighbour = [&](int ni, int nj, int nk) {
                    const double coefficient = harmonic_mean(
                        epsilon_self,
                        dielectric_at(ni, nj, nk)
                    );
                    diagonal += coefficient;
                    value -= coefficient * input[index(ni, nj, nk)];
                };
                const auto add_boundary = [&]() {
                    // Outside of the reference cube is a zero-Dirichlet boundary.
                    // A cell-centred unknown is h/2 from the boundary, so its
                    // boundary-face conductance is 2*epsilon_self.
                    diagonal += 2.0 * epsilon_self;
                };

                i > 0 ? add_neighbour(i - 1, j, k) : add_boundary();
                i + 1 < resolution_ ? add_neighbour(i + 1, j, k) : add_boundary();
                j > 0 ? add_neighbour(i, j - 1, k) : add_boundary();
                j + 1 < resolution_ ? add_neighbour(i, j + 1, k) : add_boundary();
                k > 0 ? add_neighbour(i, j, k - 1) : add_boundary();
                k + 1 < resolution_ ? add_neighbour(i, j, k + 1) : add_boundary();

                output[row] = value + diagonal * input[row];
            }
        }
    }
}

std::vector<double> GGFTSolver::build_source(KernelType kernel) const {
    std::vector<double> source(unknown_count_, 0.0);
    const double h = 2.0 / static_cast<double>(resolution_);

    if (resolution_ % 2 == 1) {
        const int c = resolution_ / 2;
        const double eps_c = dielectric_at(c, c, c);

        if (kernel == KernelType::Probability) {
            source[index(c, c, c)] = 1.0;
            return source;
        }

        int pi = c;
        int pj = c;
        int pk = c;
        int mi = c;
        int mj = c;
        int mk = c;
        if (kernel == KernelType::GradientX) {
            ++pi;
            --mi;
        } else if (kernel == KernelType::GradientY) {
            ++pj;
            --mj;
        } else {
            ++pk;
            --mk;
        }

        const double eps_plus = dielectric_at(pi, pj, pk);
        const double eps_minus = dielectric_at(mi, mj, mk);
        const double d_plus = eps_plus + eps_c;
        const double d_minus = -(eps_c + eps_minus);

        source[index(pi, pj, pk)] = eps_plus / d_plus / h;
        source[index(c, c, c)] = eps_c * (1.0 / d_plus + 1.0 / d_minus) / h;
        source[index(mi, mj, mk)] = eps_minus / d_minus / h;
        return source;
    }

    const int lower = resolution_ / 2 - 1;
    const int upper = resolution_ / 2;
    const std::array<int, 2> center_axis = {lower, upper};

    double dielectric_sum = 0.0;
    for (int k : center_axis) {
        for (int j : center_axis) {
            for (int i : center_axis) {
                dielectric_sum += dielectric_at(i, j, k);
            }
        }
    }

    if (kernel == KernelType::Probability) {
        for (int k : center_axis) {
            for (int j : center_axis) {
                for (int i : center_axis) {
                    source[index(i, j, k)] =
                        dielectric_at(i, j, k) / dielectric_sum;
                }
            }
        }
        return source;
    }

    double negative_plane_sum = 0.0;
    for (int k : center_axis) {
        for (int j : center_axis) {
            for (int i : center_axis) {
                bool negative_side = false;
                if (kernel == KernelType::GradientX) {
                    negative_side = i == lower;
                } else if (kernel == KernelType::GradientY) {
                    negative_side = j == lower;
                } else {
                    negative_side = k == lower;
                }
                if (negative_side) {
                    negative_plane_sum += dielectric_at(i, j, k);
                }
            }
        }
    }
    const double d_minus = -negative_plane_sum;

    for (int k : center_axis) {
        for (int j : center_axis) {
            for (int i : center_axis) {
                bool negative_side = false;
                if (kernel == KernelType::GradientX) {
                    negative_side = i == lower;
                } else if (kernel == KernelType::GradientY) {
                    negative_side = j == lower;
                } else {
                    negative_side = k == lower;
                }

                const double eps = dielectric_at(i, j, k);
                double coefficient = eps / dielectric_sum / h * 2.0;
                if (negative_side) {
                    coefficient = eps *
                        (1.0 / dielectric_sum + 1.0 / d_minus) /
                        h * 2.0;
                }
                source[index(i, j, k)] = coefficient;
            }
        }
    }

    return source;
}

std::vector<double> GGFTSolver::solve_source(
    const std::vector<double>& source,
    double relative_tolerance,
    int max_iterations
) const {
    if (source.size() != unknown_count_) {
        throw std::invalid_argument("solve_source(): invalid source size");
    }
    if (!(relative_tolerance > 0.0) || !std::isfinite(relative_tolerance)) {
        throw std::invalid_argument("relative_tolerance must be positive and finite");
    }
    if (max_iterations <= 0) {
        throw std::invalid_argument("max_iterations must be positive");
    }

    std::vector<double> solution(unknown_count_, 0.0);
    std::vector<double> residual = source;
    std::vector<double> direction = residual;
    std::vector<double> matrix_direction(unknown_count_, 0.0);

    double residual_squared = dot(residual, residual);
    const double source_norm = std::sqrt(residual_squared);
    if (source_norm == 0.0) {
        return solution;
    }
    const double target_norm = relative_tolerance * source_norm;
    bool converged = std::sqrt(residual_squared) <= target_norm;

    for (int iteration = 0; iteration < max_iterations && !converged; ++iteration) {
        apply_operator(direction, matrix_direction);
        const double denominator = dot(direction, matrix_direction);

        if (!std::isfinite(denominator) || denominator <= 0.0) {
            throw std::runtime_error(
                "conjugate-gradient solver encountered a non-positive denominator"
            );
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
    return solution;
}

std::vector<double> GGFTSolver::extract_boundary_flux(
    const std::vector<double>& green_values
) const {
    if (green_values.size() != unknown_count_) {
        throw std::invalid_argument("extract_boundary_flux(): invalid input size");
    }

    const std::size_t panels_per_face =
        static_cast<std::size_t>(resolution_) * static_cast<std::size_t>(resolution_);
    std::vector<double> values;
    values.reserve(6 * panels_per_face);

    const auto flux = [&](int i, int j, int k) {
        // Match the h/2 boundary distance used in add_boundary().
        return 2.0 * dielectric_at(i, j, k) * green_values[index(i, j, k)];
    };

    // z-
    for (int j = 0; j < resolution_; ++j) {
        for (int i = 0; i < resolution_; ++i) {
            values.push_back(flux(i, j, 0));
        }
    }
    // z+
    for (int j = 0; j < resolution_; ++j) {
        for (int i = 0; i < resolution_; ++i) {
            values.push_back(flux(i, j, resolution_ - 1));
        }
    }
    // y-
    for (int k = 0; k < resolution_; ++k) {
        for (int i = 0; i < resolution_; ++i) {
            values.push_back(flux(i, 0, k));
        }
    }
    // y+
    for (int k = 0; k < resolution_; ++k) {
        for (int i = 0; i < resolution_; ++i) {
            values.push_back(flux(i, resolution_ - 1, k));
        }
    }
    // x-
    for (int k = 0; k < resolution_; ++k) {
        for (int j = 0; j < resolution_; ++j) {
            values.push_back(flux(0, j, k));
        }
    }
    // x+
    for (int k = 0; k < resolution_; ++k) {
        for (int j = 0; j < resolution_; ++j) {
            values.push_back(flux(resolution_ - 1, j, k));
        }
    }

    validate_finite(values, "boundary values");
    return values;
}

std::vector<double> GGFTSolver::solve(
    KernelType kernel,
    double relative_tolerance,
    int max_iterations
) const {
    const auto source = build_source(kernel);
    const auto solution = solve_source(source, relative_tolerance, max_iterations);
    auto boundary = extract_boundary_flux(solution);

    if (kernel == KernelType::Probability) {
        constexpr double negative_tolerance = 1.0e-11;
        for (double& probability : boundary) {
            if (probability < -negative_tolerance) {
                throw std::runtime_error(
                    "solver produced a negative boundary probability"
                );
            }
            probability = std::max(0.0, probability);
        }

        const double sum = std::accumulate(boundary.begin(), boundary.end(), 0.0);
        if (!std::isfinite(sum) || sum <= 0.0) {
            throw std::runtime_error("boundary probability sum is not positive");
        }
        for (double& probability : boundary) {
            probability /= sum;
        }
    }

    return boundary;
}

GGFTResult GGFTSolver::solve_all(double relative_tolerance, int max_iterations ) const {
    GGFTResult result;
    result.resolution = resolution_;
    result.block_width = block_width_;
    result.reference_half_size = 1.0;
    result.dielectric = dielectric_;
    std::cout << "solving GFT..." << std::endl;
    result.probabilities = solve(
        KernelType::Probability,
        relative_tolerance,
        max_iterations
    );
    std::cout << "solving gradient x..." << std::endl;
    result.gradient_x = solve(
        KernelType::GradientX,
        relative_tolerance,
        max_iterations
    );
    std::cout << "solving gradient y..." << std::endl;
    result.gradient_y = solve(
        KernelType::GradientY,
        relative_tolerance,
        max_iterations
    );
    std::cout << "solving gradient z..." << std::endl;
    result.gradient_z = solve(
        KernelType::GradientZ,
        relative_tolerance,
        max_iterations
    );
    return result;
}

void GGFTSolver::write_binary( const std::string& output_path, const GGFTResult& result ) const {
    const std::size_t panel_count = 6ULL * static_cast<std::size_t>(resolution_) * static_cast<std::size_t>(resolution_);
    const std::size_t dielectric_count = dielectric_.size();

    if (result.resolution != resolution_ || result.block_width != block_width_) {
        throw std::invalid_argument("result metadata does not match solver");
    }
    if (result.dielectric.size() != dielectric_count ||
        result.probabilities.size() != panel_count ||
        result.gradient_x.size() != panel_count ||
        result.gradient_y.size() != panel_count ||
        result.gradient_z.size() != panel_count) {
        throw std::invalid_argument("result array sizes are invalid");
    }

    const std::filesystem::path path(output_path);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("could not open output file: " + output_path);
    }

    output.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    write_scalar(output, kFormatVersion);
    write_scalar(output, static_cast<std::uint32_t>(resolution_));
    write_scalar(output, static_cast<std::uint32_t>(block_width_));
    write_scalar(output, static_cast<std::uint32_t>(0));  // reserved
    write_scalar(output, result.reference_half_size);
    write_scalar(output, static_cast<std::uint64_t>(dielectric_count));
    write_scalar(output, static_cast<std::uint64_t>(panel_count));

    write_vector(output, result.dielectric);
    write_vector(output, result.probabilities);
    write_vector(output, result.gradient_x);
    write_vector(output, result.gradient_y);
    write_vector(output, result.gradient_z);

    if (!output) {
        throw std::runtime_error("failed while writing output file: " + output_path);
    }
}

}  // namespace gft
