## Bachelor Thesis:
# Non-Axis Aligned Anisotropic Certification with Randomized Smoothing

### Author: Taras Rumezhak
### Supervisor: Adel Bibi

![](images/intro_rancer.png)
Example of the l2 certificates regions presented on the
2D toy dataset, where the blue and red regions correspond to differ-
ent data classes. Orange: Data dependent isotropic region, Blue:
Anisotropic (ANCER) region, Pink: Certification region obtained
with our proposed solution - RANCER 


![](images/optimization_combined.gif)
Example of optimization procedure for different certification approaches.

Our code is implemented based on the ANCER repository (https://github.com/MotasemAlfarra/ANCER) and to run it you will need to install:

```bash
pip install ancer-python
```


### Demo Run

In our repository you can find every step described in the original thesis and the main procedures are located under the root folder however we recommend to test in with 2D version as it is easier to play with.

To run 2D pipeline:
```bash
pip install -r requirements.txt
cd pipeline_2D
python RANCER_2D
```

and to check visual comparison run:

```bash
python visualize.py
```
