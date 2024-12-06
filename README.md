```bash
cd [exmaple]
mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_C_COMPILER=hipcc -DCMAKE_PREFIX_PATH=/opt/rocm/lib/cmake
make -j
```
