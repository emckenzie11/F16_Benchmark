
# F16 Benchmark Data Analysis

The F16 Benchmark is a GVT on a F16 aircraft carrier. It's primary area of interest is surroudning the nonlinear connection between the wing and the payload. This analysis aims to identify the nonlinearities, quanitfying them where possible. Ultimately, we want to build a model that describes the system dynamics using the given training data.

See 'Nonlinear ground vibration identification of an F-16 aircraft Part I – Fast nonparametric analysis of distortions in FRF measurements' for more details about how the GVT was conducted.

See https://www.nonlinearbenchmark.org/benchmarks/f-16-gvt for the training data.

See 'F16Benchmark.pdf' for further details on the training data and the aim of this analysis.


## Pushing changes to your own repository

If you want to move the updates in this workspace to a personal remote (so you can pull them into VS Code locally), you can use Git directly:

1. Add your remote if it is not already configured:
   ```bash
   git remote add origin <your-repo-url>
   ```
2. Push the current branch (the default branch here is `work`) to that remote:
   ```bash
   git push origin work
   ```
3. On your local machine, pull from the same branch:
   ```bash
   git pull origin work
   ```

Replace `origin` with another remote name if you prefer, and ensure your remote URL points to the repository you own (for example, your GitHub fork).



