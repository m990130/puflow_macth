# sudo apt update
# sudo apt install -y libcgal-dev

mkdir -p evaluation/result


cd evaluation/evaluation_code
mkdir -p build
cd build

cmake -DCMAKE_BUILD_TYPE=Release ..
make
cp ./evaluation ../../

cd ../..

mkdir -p result

