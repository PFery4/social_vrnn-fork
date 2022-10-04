# python3 test.py --exp_num 4 --model_name SocialVRNN --num_test_sequences 100 --scenario real_world/ewap_dataset/seq_eth --record false ;
# python3 test.py --exp_num 4 --model_name SocialVRNN --num_test_sequences 10 --scenario real_world/ewap_dataset/seq_eth --record true ;

# python3 test.py --exp_num 0 --model_name SocialVRNN --num_test_sequences 100 --scenario real_world/ewap_dataset/seq_hotel --record false ;
# python3 test.py --exp_num 0 --model_name SocialVRNN --num_test_sequences 10 --scenario real_world/ewap_dataset/seq_hotel --record true ;

# python3 test.py --exp_num 1 --model_name SocialVRNN --num_test_sequences 100 --scenario real_world/zara_01 --record false ;
# python3 test.py --exp_num 1 --model_name SocialVRNN --num_test_sequences 10 --scenario real_world/zara_01 --record true ;

# python3 test.py --exp_num 2 --model_name SocialVRNN --num_test_sequences 100 --scenario real_world/st --record false ;
# python3 test.py --exp_num 2 --model_name SocialVRNN --num_test_sequences 10 --scenario real_world/st --record true ;

# python3 test.py --exp_num 3 --model_name SocialVRNN --num_test_sequences 100 --scenario real_world/zara_02 --record false ;
# python3 test.py --exp_num 3 --model_name SocialVRNN --num_test_sequences 10 --scenario real_world/zara_02 --record true ;


# TESTING SECTION, VERIFYING THAT THE TEST SCRIPT WORKS PROPERLY
#python3 test.py --exp_num 4000 --model_name SocialVRNN --num_test_sequences 100 --scenario real_world/ewap_dataset/seq_hotel --record false ; # DONE
#python3 test.py --exp_num 4000 --model_name SocialVRNN --num_test_sequences 10 --scenario real_world/ewap_dataset/seq_hotel --record true ; # DONE

# python3 test.py --exp_num 4001 --model_name SocialVRNN --num_test_sequences 100 --scenario real_world/ewap_dataset/seq_hotel --record false ; # DONE
# python3 test.py --exp_num 4001 --model_name SocialVRNN --num_test_sequences 10 --scenario real_world/ewap_dataset/seq_hotel --record true ; # DONE

for i in {1..10}
do
    python3 test.py --exp_num 5000 --model_name SocialVRNN --num_test_sequences 100 --scenario real_world/ewap_dataset/seq_hotel --record false ;
    python3 test.py --exp_num 5000 --model_name SocialVRNN --num_test_sequences 10 --scenario real_world/ewap_dataset/seq_hotel --record true ;
done


for i in {1..10}
do
    python3 test.py --exp_num 5001 --model_name SocialVRNN --num_test_sequences 100 --scenario real_world/ewap_dataset/seq_hotel --record false ;
    python3 test.py --exp_num 5001 --model_name SocialVRNN --num_test_sequences 10 --scenario real_world/ewap_dataset/seq_hotel --record true ;
done


for i in {1..10}
do
    python3 test.py --exp_num 5002 --model_name SocialVRNN --num_test_sequences 100 --scenario real_world/ewap_dataset/seq_eth --record false ;
    python3 test.py --exp_num 5002 --model_name SocialVRNN --num_test_sequences 10 --scenario real_world/ewap_dataset/seq_eth --record true ;
done


for i in {1..10}
do
    python3 test.py --exp_num 5003 --model_name SocialVRNN --num_test_sequences 100 --scenario real_world/ewap_dataset/seq_eth --record false ;
    python3 test.py --exp_num 5003 --model_name SocialVRNN --num_test_sequences 10 --scenario real_world/ewap_dataset/seq_eth --record true ;
done


for i in {1..10}
do
    python3 test.py --exp_num 5004 --model_name SocialVRNN --num_test_sequences 100 --scenario real_world/st --record false ;
    python3 test.py --exp_num 5004 --model_name SocialVRNN --num_test_sequences 10 --scenario real_world/st --record true ;
done


for i in {1..10}
do
    python3 test.py --exp_num 5005 --model_name SocialVRNN --num_test_sequences 100 --scenario real_world/st --record false ;
    python3 test.py --exp_num 5005 --model_name SocialVRNN --num_test_sequences 10 --scenario real_world/st --record true ;
done

for i in {1..10}
do
    python3 test.py --exp_num 5006 --model_name SocialVRNN --num_test_sequences 100 --scenario real_world/zara_01 --record false ;
    python3 test.py --exp_num 5006 --model_name SocialVRNN --num_test_sequences 10 --scenario real_world/zara_01 --record true ;
done

for i in {1..10}
do
    python3 test.py --exp_num 5007 --model_name SocialVRNN --num_test_sequences 100 --scenario real_world/zara_01 --record false ;
    python3 test.py --exp_num 5007 --model_name SocialVRNN --num_test_sequences 10 --scenario real_world/zara_01 --record true ;
done

for i in {1..10}
do
    python3 test.py --exp_num 5008 --model_name SocialVRNN --num_test_sequences 100 --scenario real_world/zara_02 --record false ;
    python3 test.py --exp_num 5008 --model_name SocialVRNN --num_test_sequences 10 --scenario real_world/zara_02 --record true ;
done

for i in {1..10}
do
    python3 test.py --exp_num 5009 --model_name SocialVRNN --num_test_sequences 100 --scenario real_world/zara_02 --record false ;
    python3 test.py --exp_num 5009 --model_name SocialVRNN --num_test_sequences 10 --scenario real_world/zara_02 --record true ;
done




