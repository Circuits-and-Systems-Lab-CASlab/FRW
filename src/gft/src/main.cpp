#include "ggft_solver.h"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>

namespace {

void print_usage(const char* program) {
    std::cerr
        << "Usage: " << program
        << " <output.bin> [resolution=23] [relative_tolerance=1e-12]"
        << " [max_iterations=20000]\n";
}

}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const std::string output_path = argv[1];
    const int resolution = argc >= 3 ? std::stoi(argv[2]) : 23;
    const double tolerance = argc >= 4 ? std::stod(argv[3]) : 1.0e-12;
    const int max_iterations = argc >= 5 ? std::stoi(argv[4]) : 20000;

    const gft::HomogeneousCubeGFTSolver solver(resolution);
    const auto probabilities = solver.solve(tolerance, max_iterations);
    solver.write_deeprwcap_binary(output_path, probabilities);

    const std::size_t panels_per_face = static_cast<std::size_t>(resolution) * static_cast<std::size_t>(resolution);

    std::cout << std::setprecision(16);
    std::cout << "Generated homogeneous cube GFT\n";
    std::cout << "Resolution: " << resolution << '\n';
    std::cout << "Panels: " << probabilities.size() << '\n';
    std::cout << "Probability sum: " << std::accumulate(probabilities.begin(), probabilities.end(), 0.0) << '\n';

    static const char* face_names[] = {"z-", "z+", "y-", "y+", "x-", "x+"};
    for (std::size_t face = 0; face < 6; ++face) {
        const auto begin = probabilities.begin() +
            static_cast<std::ptrdiff_t>(face * panels_per_face);
        const auto end = begin + static_cast<std::ptrdiff_t>(panels_per_face);
        std::cout << face_names[face] << ": "
                    << std::accumulate(begin, end, 0.0) << '\n';
    }

    std::cout << "Binary: " << output_path << '\n';
    return EXIT_SUCCESS;
}
