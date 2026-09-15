#include "ggft_solver.h"

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string output_path;
    int resolution = 23;
    int block_width = 1;
    double tolerance = 1.0e-10;
    int max_iterations = 20000;
    std::string dielectric_path;
};

void print_usage(const char* program) {
    std::cerr
        << "Usage: " << program << " --output <tables.bin> [options]\n\n"
        << "Options:\n"
        << "  --resolution <N>          Grid resolution (default: 23)\n"
        << "  --block-width <W>         Dielectric block width (default: 1)\n"
        << "  --tolerance <value>       Relative CG tolerance (default: 1e-10)\n"
        << "  --max-iterations <count>  CG iteration limit (default: 20000)\n"
        << "  --dielectric <raw.bin>    Raw float64 epsilon_r array; default is all 1\n"
        << "  --help                    Show this message\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        const auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else if (argument == "--output" || argument == "-o") {
            options.output_path = require_value("--output");
        } else if (argument == "--resolution") {
            options.resolution = std::stoi(require_value("--resolution"));
        } else if (argument == "--block-width") {
            options.block_width = std::stoi(require_value("--block-width"));
        } else if (argument == "--tolerance") {
            options.tolerance = std::stod(require_value("--tolerance"));
        } else if (argument == "--max-iterations") {
            options.max_iterations = std::stoi(require_value("--max-iterations"));
        } else if (argument == "--dielectric") {
            options.dielectric_path = require_value("--dielectric");
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }

    if (options.output_path.empty()) {
        throw std::invalid_argument("--output is required");
    }
    return options;
}

std::vector<double> read_dielectric_file(
    const std::string& path,
    std::size_t expected_count
) {
    if (path.empty()) {
        return {};
    }

    const std::uintmax_t expected_bytes = expected_count * sizeof(double);
    const std::uintmax_t actual_bytes = std::filesystem::file_size(path);
    if (actual_bytes != expected_bytes) {
        throw std::runtime_error(
            "dielectric file has " + std::to_string(actual_bytes) +
            " bytes; expected " + std::to_string(expected_bytes)
        );
    }

    std::vector<double> values(expected_count);
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open dielectric file: " + path);
    }
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(expected_bytes)
    );
    if (!input) {
        throw std::runtime_error("failed while reading dielectric file: " + path);
    }
    return values;
}

double sum(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.block_width <= 0 || options.resolution % options.block_width != 0) {
            throw std::invalid_argument( "block width must be positive and divide resolution");
        }
        if (options.resolution % 2 == 0) {
            std::cerr
                << "Warning: an even resolution has no cell centred exactly at "
                << "the reference-cube source. Prefer an odd resolution (for "
                << "example 201 instead of 200).\n";
        }

        const int dielectric_resolution = options.resolution / options.block_width;
        const std::size_t dielectric_count =
            static_cast<std::size_t>(dielectric_resolution) *
            static_cast<std::size_t>(dielectric_resolution) *
            static_cast<std::size_t>(dielectric_resolution);
        auto dielectric = read_dielectric_file( options.dielectric_path, dielectric_count );

        //init
        const gft::GGFTSolver solver(options.resolution, options.block_width, std::move(dielectric) );

        //solve for GFT, DP/dx, DP/dy, DP/dz
        const auto result = solver.solve_all( options.tolerance, options.max_iterations );

        solver.write_binary(options.output_path, result);

        std::cout << std::setprecision(16);
        std::cout << "Generated GGFT reference-cube tables\n";
        std::cout << "Resolution: " << result.resolution << '\n';
        std::cout << "Block width: " << result.block_width << '\n';
        std::cout << "Dielectric cells: " << result.dielectric.size() << '\n';
        std::cout << "Boundary panels: " << result.probabilities.size() << '\n';
        std::cout << "P sum: " << sum(result.probabilities) << '\n';
        std::cout << "dP/dx sum: " << sum(result.gradient_x) << '\n';
        std::cout << "dP/dy sum: " << sum(result.gradient_y) << '\n';
        std::cout << "dP/dz sum: " << sum(result.gradient_z) << '\n';
        std::cout << "Binary: " << options.output_path << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
}
