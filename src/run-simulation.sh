


# for mode in sequential simultaneous
# do
# 	for graph in Email Facebook BTER SBM
# 	do
# 		for var in 0.001 0.01 0.1 0.5
# 		do
# 			for i in {1..99}
# 			do
# 				python youtubeSim.py --b_mode=uniform --beta_mode=control-homophily  --graph=$graph --seed=$i --control_var=$var >> ../result/${mode}/control-homophily/b_uniform_${graph}_control-homophily_${var}.txt &
# 			done
# 			BACK_PID=$!
# 			wait $BACK_PID
# 		done
# 	done
# done


# ##########################################################################################

# for mode in sequential simultaneous
# do
# 	for graph in Email Facebook BTER SBM
# 	do
# 		for i in {1..99}
# 		do
# 		      python youtubeSim.py --output=1 --mode=$mode  --b_mode=uniform --beta_mode=homophily  --graph=$graph --seed=$i  >> ../result/${mode}/b_uniform_${graph}_homophily_stats.txt &
# 		      python youtubeSim.py --output=1 --mode=$mode --b_mode=uniform --beta_mode=gaussian  --graph=$graph --seed=$i  >> ../result/${mode}/b_uniform_${graph}_gaussian_stats.txt &
# 		      python youtubeSim.py --output=1 --mode=$mode --b_mode=uniform --beta_mode=fully-homophily  --graph=$graph --seed=$i  >> ../result/${mode}/b_uniform_${graph}_fully-homophily_stats.txt &
# 		done
# 	  	BACK_PID=$!
# 	    wait $BACK_PID
# 	done
# done

# ##########################################################################################


# for mode in sequential simultaneous
# do
# 	for graph in Email Facebook BTER SBM
# 	do
# 		for i in {1..99}
# 		do
# 		      python youtubeSim.py --output=0 --mode=$mode  --b_mode=uniform --beta_mode=homophily  --graph=$graph --seed=$i  >> ../result/${mode}/b_uniform_${graph}_homophily.txt &
# 		      python youtubeSim.py --output=0 --mode=$mode  --b_mode=uniform --beta_mode=gaussian  --graph=$graph --seed=$i  >> ../result/${mode}/b_uniform_${graph}_gaussian.txt &
# 		      python youtubeSim.py --output=0 --mode=$mode  --b_mode=uniform --beta_mode=fully-homophily  --graph=$graph --seed=$i  >> ../result/${mode}/b_uniform_${graph}_fully-homophily.txt &
# 		done
# 	  	BACK_PID=$!
# 	    wait $BACK_PID
# 	done
# done


# ##########################################################################################
# for mode in sequential simultaneous
# do
# 	for graph in Email Facebook SBM BTER
# 	do
# 		for i in {1..50}
# 		do				
# 			if [ $graph == Facebook ]
# 			then
# 			   maxIter=6000
# 			elif [ $graph == Email ]
# 			then 
# 				maxIter=3000
# 			elif [ $graph == SBM ]
# 			then 
# 				maxIter=1500
# 			else
# 			   maxIter=1000
# 			fi
# 			python bestshot.py --graph=$graph --seed=$i --maxIter=$maxIter   >> ../result/${mode}/bestshot_${graph}.txt &
# 		done
# 		BACK_PID=$!
# 		wait $BACK_PID
# 	done
# done

# # ##########################################################################################
# # maxIter=200
# # for graph in Email
# # do
# # 	for i in {1..50}
# # 	do
		
# # 		python discrete.py --graph=$graph --seed=$i --maxIter=3000   >> ../result/bestshot_${graph}.txt &
# # 		# python discrete.py --graph=$graph --seed=$i --maxIter=$maxIter   >> ../result/bestshot_${graph}.txt &
# # 		# BACK_PID=$!
# # 		# wait $BACK_PID
# # 	done
# # 		BACK_PID=$!
# # 		wait $BACK_PID
# # done

##########################################################################################
for graph in SBM BTER
do
	for i in {1..30}
	do
		
		python gcnCO.py --graph=$graph --seed=$i  >> ../result/gcn_${graph}.txt &
		# python discrete.py --graph=$graph --seed=$i --maxIter=$maxIter   >> ../result/bestshot_${graph}.txt &
		# BACK_PID=$!
		# wait $BACK_PID
	done
	BACK_PID=$!
	wait $BACK_PID
done
