python3 ../crossvalidate.py lenet5 10 --learning_rate=0.01 --momentum=0.9 --batch_size=128 > lenet-lr01.txt &
python3 ../crossvalidate.py lenet5 10 --learning_rate=0.05 --momentum=0.9 --batch_size=128 > lenet-lr04.txt &
python3 ../crossvalidate.py lenet5 10 --learning_rate=0.08 --momentum=0.9 --batch_size=128 > lenet-lr07.txt &
python3 ../crossvalidate.py lenet5 10 --learning_rate=0.1 --momentum=0.9 --batch_size=128 > lenet-lr10.txt &

python3 ../crossvalidate.py basic 10 --learning_rate=0.01 --momentum=0.9 --batch_size=128 > basic-lr01.txt
python3 ../crossvalidate.py basic 10 --learning_rate=0.05 --momentum=0.9 --batch_size=128 > basic-lr04.txt
python3 ../crossvalidate.py basic 10 --learning_rate=0.08 --momentum=0.9 --batch_size=128 > basic-lr07.txt
python3 ../crossvalidate.py basic 10 --learning_rate=0.1 --momentum=0.9 --batch_size=128 > basic-lr10.txt