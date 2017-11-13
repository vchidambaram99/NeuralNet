This is my second attempt to create some neural network implementations in C++ (tensorflow and all other libraries you could use are probably better, this is mostly a project for me to learn from).
It relies on the linear algebra library Eigen, so that must be set up for this to work.
Also, when compiling this, you must have optimizations turned on (-O3 flag for g++), otherwise it is painfully slow.
This is also currently very broken on the Layers branch, but it will be fixed eventually (it also overhauls the thing almost entirely).
