#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace gft {

class MultiDielectricCubeGFTSolver {
public:
    explicit MultiDielectricCubeGFTSolver(
        int resolution,
        double interface_height,
        double eps_bottom,
        double eps_top
    );

    [[nodiscard]] int resolution() const noexcept;

    // Solve the discrete Green-function system for a source at the cube centre.
    // Returns the 6*N*N boundary probabilities in this face order:
    // z-, z+, y-, y+, x-, x+.
    [[nodiscard]] std::vector<double> solve(
        double relative_tolerance = 1.0e-12,
        int max_iterations = 20000
    ) const;

    // Write a DeepRWCap-style binary:
    // double N, double block_width, N^3 actual dielectric values, 6*N^2 GFT values.
    void write_deeprwcap_binary(
        const std::string& output_path,
        const std::vector<double>& probabilities
    ) const;

private:
    int resolution_;
    double interface_height_;
    double eps_bottom_;
    double eps_top_;
    std::size_t unknown_count_;

    [[nodiscard]] std::size_t index(int i, int j, int k) const noexcept;
    
    // Helper to get permittivity at a specific Z cell index
    [[nodiscard]] double get_cell_eps(int k) const noexcept;
    
    // Helper to get effective permittivity between two adjacent Z cells (harmonic mean)
    [[nodiscard]] double get_edge_eps(int k1, int k2) const noexcept;

    void apply_laplacian(
        const std::vector<double>& input,
        std::vector<double>& output
    ) const;
    
    [[nodiscard]] std::vector<double> build_centre_source() const;
    [[nodiscard]] std::vector<double> extract_boundary_probabilities(
        const std::vector<double>& green_values
    ) const;
};

} // namespace gft