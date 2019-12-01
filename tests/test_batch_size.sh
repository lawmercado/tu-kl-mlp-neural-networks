python3 ../crossvalidate.py lenet5 10 --learning_rate=0.01 --momentum=0.85 --batch_size=64 > lenet-bs64.txt &
python3 ../crossvalidate.py lenet5 10 --learning_rate=0.01 --momentum=0.85 --batch_size=100 > lenet-bs100.txt &
python3 ../crossvalidate.py lenet5 10 --learning_rate=0.01 --momentum=0.85 --batch_size=128 > lenet-bs128.txt &

python3 ../crossvalidate.py basic 10 --learning_rate=0.01 --momentum=0.5 --batch_size=64 > basic-bs64.txt
python3 ../crossvalidate.py basic 10 --learning_rate=0.01 --momentum=0.5 --batch_size=100 > basic-bs100.txt
python3 ../crossvalidate.py basic 10 --learning_rate=0.01 --momentum=0.5 --batch_size=128 > basic-bs128.txt
