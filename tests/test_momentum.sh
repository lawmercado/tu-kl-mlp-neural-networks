python3 ../crossvalidate.py lenet5 10 --learning_rate=0.05 --momentum=0.5 --batch_size=128 > lenet-mom05.txt &
python3 ../crossvalidate.py lenet5 10 --learning_rate=0.05 --momentum=0.75 --batch_size=128 > lenet-mom075.txt &
python3 ../crossvalidate.py lenet5 10 --learning_rate=0.05 --momentum=0.85 --batch_size=128 > lenet-mom085.txt &

python3 ../crossvalidate.py basic 10 --learning_rate=0.05 --momentum=0.5 --batch_size=128 > basic-mom05.txt
python3 ../crossvalidate.py basic 10 --learning_rate=0.05 --momentum=0.75 --batch_size=128 > basic-mom075.txt
python3 ../crossvalidate.py basic 10 --learning_rate=0.05 --momentum=0.85 --batch_size=128 > basic-mom085.txt
