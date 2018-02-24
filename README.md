This is my second attempt to create some neural network implementations in C++ (tensorflow and all other libraries you could use are probably better, this is mostly a project for me to learn from).
It relies on the linear algebra library Eigen, so that must be set up for this to work.
Also, when compiling this, you must have optimizations turned on (-O3 flag for g++), otherwise it is painfully slow.
This is also currently very broken on the Layers branch, but it will be fixed eventually (it also overhauls the thing almost entirely).

This project will also probably be on hold until I finish more of my tensor library so that I can implement neural nets using it. My original plan was to use Eigen's Tensor module, but it needs tensor rank as a template parameter, which makes it more difficult to use.
