This project implements a Python emulator of NeuroMem's CM1K neuromorphic chip. It should, in theory, produce nearly identical internal models (neural networks) and subsequent predictions to those of the hardware chip itself. However, I do not yet personally have access to a CM1K chip and so cannot make such a comparison.

The emulator currently includes drivers for four popular public datasets, described below. It should be possible to extrapolate these drivers to support other datasets. Note that the drivers are not designed toward creating a tool for using the emulator for real world modeling and prediction, but rather are designed for running experiments on the CM1K's modeling performance relative to various settings and gathering the results into text dumps that can be (somewhat) easily moved to a spreadsheet.

The main driver is test.py. It contains __main__, main(), and runners for the four public datasets that are currently supported. It also holds the unit tests, such as they are.

The following four files encapsulate reading four popular public datasets:
- mnist.py
- att_db_of_faces.py
- iris.py
- mushroom.py
The top of each file provides information about its dataset, including where you can download them. The datasets are not bundled with this project. After you download them, you will have to configure the paths to the datasets on your local filesystem so the runners (see above) can find them. These paths are specified in module-level variables at the top of each of the four files.

The following two files represent the CM1K emulator:
- cm1k_emulator.py
- neuron.py
The emulator can be used in loosely two ways, low level emulation and high level emulation. Low level emulation attempts to replicate the chip's own internal behavior at a very precise granularity. For example, it processes individual bytes of the input pattern one by one, broadcasting those bytes to the neurons and allowing the neurons to incrementally update their internally maintained distance metrics one input byte at a time. This is how the real CM1K chip works, each neuron steadily updating its distance as each new input byte is received. There are other examples of low-level behavior too. The high level approach should be much more efficient (faster) without altering the modeling or prediction results. For example, when presenting an input pattern to the neurons, it doesn't iterate over the bytes of the input one by one, but rather passes the input array whole-parcel to the neurons, which then calculate the total pattern distance (i.e., pattern-to-pattern difference) in one step.

20160611
Keith Wiley
kwiley@keithwiley.com
http://keithwiley.com
http://keithwiley.com/software/CM1K_emulator.shtml
https://github.com/kebwi/CM1K_emulator
