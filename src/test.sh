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

# for i in {1..10}
# do
#     python3 test.py --exp_num 100 --model_name SocialVRNN_LSTM_ED --num_test_sequences 100 --scenario "real_world/ewap_dataset/seq_hotel" --record false;
#     python3 test.py --exp_num 101 --model_name SocialVRNN_LSTM_ED --num_test_sequences 100 --scenario "real_world/ewap_dataset/seq_eth" --record false;
#     python3 test.py --exp_num 102 --model_name SocialVRNN_LSTM_ED --num_test_sequences 100 --scenario "real_world/st" --record false;
#     python3 test.py --exp_num 200 --model_name SocialVRNN_LSTM_ED --num_test_sequences 100 --scenario "real_world/ewap_dataset/seq_hotel" --record false;
#     python3 test.py --exp_num 201 --model_name SocialVRNN_LSTM_ED --num_test_sequences 100 --scenario "real_world/ewap_dataset/seq_eth" --record false;
#     python3 test.py --exp_num 202 --model_name SocialVRNN_LSTM_ED --num_test_sequences 100 --scenario "real_world/st" --record false;
#     python3 test.py --exp_num 300 --model_name SocialVRNN_LSTM_ED --num_test_sequences 100 --scenario "real_world/ewap_dataset/seq_hotel" --record false;
#     python3 test.py --exp_num 301 --model_name SocialVRNN_LSTM_ED --num_test_sequences 100 --scenario "real_world/ewap_dataset/seq_eth" --record false;
#     python3 test.py --exp_num 302 --model_name SocialVRNN_LSTM_ED --num_test_sequences 100 --scenario "real_world/st" --record false;
#     python3 test.py --exp_num 400 --model_name SocialVRNN_LSTM_ED --num_test_sequences 100 --scenario "real_world/ewap_dataset/seq_hotel" --record false;
#     python3 test.py --exp_num 401 --model_name SocialVRNN_LSTM_ED --num_test_sequences 100 --scenario "real_world/ewap_dataset/seq_eth" --record false;
#     python3 test.py --exp_num 402 --model_name SocialVRNN_LSTM_ED --num_test_sequences 100 --scenario "real_world/st" --record false;
#    python3 test.py --exp_num 103 --model_name SocialVRNN_LSTM_ED --num_test_sequences 100 --scenario "real_world/zara_01" --record false;
#    python3 test.py --exp_num 104 --model_name SocialVRNN_LSTM_ED --num_test_sequences 100 --scenario "real_world/zara_02" --record false;
    
#    python3 test.py --exp_num 203 --model_name SocialVRNN_LSTM_ED --num_test_sequences 100 --scenario "real_world/zara_01" --record false;
#    python3 test.py --exp_num 204 --model_name SocialVRNN_LSTM_ED --num_test_sequences 100 --scenario "real_world/zara_02" --record false;
    
#    python3 test.py --exp_num 303 --model_name SocialVRNN_LSTM_ED --num_test_sequences 100 --scenario "real_world/zara_01" --record false;
#    python3 test.py --exp_num 304 --model_name SocialVRNN_LSTM_ED --num_test_sequences 100 --scenario "real_world/zara_02" --record false;
    
#    python3 test.py --exp_num 403 --model_name SocialVRNN_LSTM_ED --num_test_sequences 100 --scenario "real_world/zara_01" --record false;
#    python3 test.py --exp_num 404 --model_name SocialVRNN_LSTM_ED --num_test_sequences 100 --scenario "real_world/zara_02" --record false;
# done

# python3 test.py --exp_num 100 --model_name SocialVRNN_LSTM_ED --num_test_sequences 10 --scenario "real_world/ewap_dataset/seq_hotel" --record true;
# python3 test.py --exp_num 101 --model_name SocialVRNN_LSTM_ED --num_test_sequences 10 --scenario "real_world/ewap_dataset/seq_eth" --record true;
# python3 test.py --exp_num 102 --model_name SocialVRNN_LSTM_ED --num_test_sequences 10 --scenario "real_world/st" --record true;
# python3 test.py --exp_num 200 --model_name SocialVRNN_LSTM_ED --num_test_sequences 10 --scenario "real_world/ewap_dataset/seq_hotel" --record true;
# python3 test.py --exp_num 201 --model_name SocialVRNN_LSTM_ED --num_test_sequences 10 --scenario "real_world/ewap_dataset/seq_eth" --record true;
# python3 test.py --exp_num 202 --model_name SocialVRNN_LSTM_ED --num_test_sequences 10 --scenario "real_world/st" --record true;
# python3 test.py --exp_num 300 --model_name SocialVRNN_LSTM_ED --num_test_sequences 10 --scenario "real_world/ewap_dataset/seq_hotel" --record true;
# python3 test.py --exp_num 301 --model_name SocialVRNN_LSTM_ED --num_test_sequences 10 --scenario "real_world/ewap_dataset/seq_eth" --record true;
# python3 test.py --exp_num 302 --model_name SocialVRNN_LSTM_ED --num_test_sequences 10 --scenario "real_world/st" --record true;
# python3 test.py --exp_num 400 --model_name SocialVRNN_LSTM_ED --num_test_sequences 10 --scenario "real_world/ewap_dataset/seq_hotel" --record true;
# python3 test.py --exp_num 401 --model_name SocialVRNN_LSTM_ED --num_test_sequences 10 --scenario "real_world/ewap_dataset/seq_eth" --record true;
# python3 test.py --exp_num 402 --model_name SocialVRNN_LSTM_ED --num_test_sequences 10 --scenario "real_world/st" --record true;
# python3 test.py --exp_num 103 --model_name SocialVRNN_LSTM_ED --num_test_sequences 10 --scenario "real_world/zara_01" --record true;
# python3 test.py --exp_num 104 --model_name SocialVRNN_LSTM_ED --num_test_sequences 10 --scenario "real_world/zara_02" --record true;
# python3 test.py --exp_num 203 --model_name SocialVRNN_LSTM_ED --num_test_sequences 10 --scenario "real_world/zara_01" --record true;
# python3 test.py --exp_num 204 --model_name SocialVRNN_LSTM_ED --num_test_sequences 10 --scenario "real_world/zara_02" --record true;
# python3 test.py --exp_num 303 --model_name SocialVRNN_LSTM_ED --num_test_sequences 10 --scenario "real_world/zara_01" --record true;
# python3 test.py --exp_num 304 --model_name SocialVRNN_LSTM_ED --num_test_sequences 10 --scenario "real_world/zara_02" --record true;
# python3 test.py --exp_num 403 --model_name SocialVRNN_LSTM_ED --num_test_sequences 10 --scenario "real_world/zara_01" --record true;
# python3 test.py --exp_num 404 --model_name SocialVRNN_LSTM_ED --num_test_sequences 10 --scenario "real_world/zara_02" --record true;

python3 test.py --exp_num 200 --model_name SocialVRNN_LSTM_ED --num_test_sequences 100 --scenario "real_world/ewap_dataset/seq_hotel" --record false;
python3 test.py --exp_num 201 --model_name SocialVRNN_LSTM_ED --num_test_sequences 100 --scenario "real_world/ewap_dataset/seq_eth" --record false;
python3 test.py --exp_num 202 --model_name SocialVRNN_LSTM_ED --num_test_sequences 100 --scenario "real_world/st" --record false;
python3 test.py --exp_num 203 --model_name SocialVRNN_LSTM_ED --num_test_sequences 100 --scenario "real_world/zara_01" --record false;
python3 test.py --exp_num 204 --model_name SocialVRNN_LSTM_ED --num_test_sequences 100 --scenario "real_world/zara_02" --record false;

