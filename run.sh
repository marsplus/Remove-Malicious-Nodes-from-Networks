
nCPU=30
nExp=30
for i in  2 3
do
    python src/main.py --graph_type=BA --batch=$i --numExp=$nExp --numCPU=$nCPU 
    python src/main.py --graph_type=Small-World --batch=$i --numExp=$nExp --numCPU=$nCPU 
    #python src/main.py --graph_type=Facebook --batch=$i --numExp=$nExp --numCPU=$nCPU 
done


#python src/main.py Small-World 1 50
#python src/main.py BA 2 50
#python src/main.py Small-World 2 50
#python src/main.py BA 3 50
#python src/main.py Small-World 3 50
#python src/main.py Facebook 1 50
