#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gft {

enum class KernelType {
    Probability,
    GradientX,
    GradientY,
    GradientZ,
};

struct GGFTResult {
    int resolution = 0;
    int block_width = 1;
    double reference_half_size = 1.0;
    std::vector<double> dielectric;
    std::vector<double> probabilities;
    std::vector<double> gradient_x;
    std::vector<double> gradient_y;
    std::vector<double> gradient_z;
};

class GGFTSolver {
public:
    GGFTSolver(
        int resolution,
        int block_width = 1,
        std::vector<double> dielectric = {}
    );

    int resolution() const;
    int block_width() const;
    int dielectric_resolution() const;
    const std::vector<double>& dielectric() const;

    // Solve the reference cube [-1, 1]^3. Boundary-panel order:
    // z-, z+, y-, y+, x-, x+.
    GGFTResult solve_all( double relative_tolerance = 1.0e-10, int max_iterations = 20000 ) const;

    std::vector<double> solve(
        KernelType kernel,
        double relative_tolerance = 1.0e-10,
        int max_iterations = 20000
    ) const;

    // Versioned deterministic bundle:
    // header + dielectric + P + dP/dxi_x + dP/dxi_y + dP/dxi_z.
    void write_binary( const std::string& output_path, const GGFTResult& result ) const;

private:
    int resolution_;
    int block_width_;
    int dielectric_resolution_;
    std::size_t unknown_count_;
    std::vector<double> dielectric_;

    std::size_t index(int i, int j, int k) const;
    std::size_t dielectric_index(int i, int j, int k) const;
    double dielectric_at(int i, int j, int k) const;

    void apply_operator(
        const std::vector<double>& input,
        std::vector<double>& output
    ) const;

    std::vector<double> build_source(KernelType kernel) const;
    std::vector<double> solve_source(
        const std::vector<double>& source,
        double relative_tolerance,
        int max_iterations
    ) const;
    std::vector<double> extract_boundary_flux(
        const std::vector<double>& green_values
    ) const;
};

}  // namespace gft
