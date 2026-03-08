# Scientific Computing: DLA Fractal Growth and Reaction-Diffusion Models

This project explores two different fractal growth models—Diffusion-Limited Aggregation (DLA)—and the Gray-Scott reaction-diffusion system.

Two methods for DLA simulation are implemented:
1.  **Monte Carlo (MC) Method**: Based on random-walking particles that stick to a growing cluster with a certain "sticking probability" `p_s` upon contact.
2.  **Partial Differential Equation (PDE) Method**: Models the particle concentration field as a Laplace equation. The solution of this equation determines where the cluster grows.

Additionally, the project simulates the Gray-Scott reaction-diffusion model, which can generate a variety of complex, life-like patterns.

<ul>
  <li>Fig. 1: DLA_PDE_examples.py</li>
  <li>Fig. 2: optimal_omega_search.py (<em>Note: runs on all available cores, taking approximately 1 hour to complete on an M1 macbook air</em>).</li>
  <li>Fig. 3: first output of PDE_vs_MC_comparison1.py</li>
  <li>Fig. 4 & 5: second output of PDE_vs_MC_comparison1.py (<em>Note: takes approximately 10 minutes to complete on an M2 macbook air</em></li>
  <li>Fig. 6 & 7: DLA_MC_statistics.py</li>
  <li>Fig. 8: DLA_vs_MC_comparison2.py (<em>Note: runs on all available cores, taking approximately 2 minutes to complete on an M1 macbook air</em>)</li>
  <li>Fig. 9: Gray_Scott_phase.py (<em>Note: runs as an infinite animation, taking approximately 10 minutes to reach the timepoint shown in the report</em>)</li>
  <li>Fig. 10: Gray_Scott.py (<em>Note: runs as an infinite animation, for the different cases modify line 27: k, f = A, B or C</em>)</li>
</ul>

