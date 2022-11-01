
seed=29
for graph in Youtube
do
    for nComm in 500  
    do
        python youtubeSim.py --b_mode=uniform --beta_mode=homophily  --graph=$graph --seed=$seed --nComm=$nComm  &
        python youtubeSim.py --b_mode=uniform --beta_mode=gaussian  --graph=$graph --seed=$seed --nComm=$nComm   &
        python youtubeSim.py --b_mode=uniform --beta_mode=uniform  --graph=$graph --seed=$seed --nComm=$nComm    & 
        python youtubeSim.py --b_mode=uniform --beta_mode=fully-homophily  --graph=$graph --seed=$seed --nComm=$nComm &
    done
    BACK_PID=$! 
    wait $BACK_PID
done
