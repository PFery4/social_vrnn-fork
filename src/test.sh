python3 test.py --exp_num 4 --model_name SocialVRNN --num_test_sequences 100 --scenario real_world/ewap_dataset/seq_eth --record false ;
python3 test.py --exp_num 4 --model_name SocialVRNN --num_test_sequences 10 --scenario real_world/ewap_dataset/seq_eth --record true ;

python3 test.py --exp_num 0 --model_name SocialVRNN --num_test_sequences 100 --scenario real_world/ewap_dataset/seq_hotel --record false ;
python3 test.py --exp_num 0 --model_name SocialVRNN --num_test_sequences 10 --scenario real_world/ewap_dataset/seq_hotel --record true ;

python3 test.py --exp_num 1 --model_name SocialVRNN --num_test_sequences 100 --scenario real_world/zara_01 --record false ;
python3 test.py --exp_num 1 --model_name SocialVRNN --num_test_sequences 10 --scenario real_world/zara_01 --record true ;

python3 test.py --exp_num 2 --model_name SocialVRNN --num_test_sequences 100 --scenario real_world/st --record false ;
python3 test.py --exp_num 2 --model_name SocialVRNN --num_test_sequences 10 --scenario real_world/st --record true ;

python3 test.py --exp_num 3 --model_name SocialVRNN --num_test_sequences 100 --scenario real_world/zara_02 --record false ;
python3 test.py --exp_num 3 --model_name SocialVRNN --num_test_sequences 10 --scenario real_world/zara_02 --record true ;
