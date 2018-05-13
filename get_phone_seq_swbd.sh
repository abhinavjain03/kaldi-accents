. ./cmd.sh
. ./path.sh

nj=20
modelDirectory=exp/alreadyTrainedModelsOnSwbd
graph_dir=$modelDirectory/tri4/graph_sw1_tg
dir=$modelDirectory/tdnn_d_sp



x=accent_id_cslu_data

. utils/parse_options.sh

mfcc=0
mfcchires=0
ivector=0
decode=0

decodeWithoutLM=1

phones=1

if [ $mfcc -eq 1 ]; then

  	mfccdir=exp/mfcc
	steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc.conf --cmd "$train_cmd" \
			data/$x exp/make_mfcc/$x $mfccdir
	steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
	utils/fix_data_dir.sh data/$x


fi

if [ $mfcchires -eq 1 ]; then
	mfccdir=exp/mfcc_hires
	utils/copy_data_dir.sh data/$x data/${x}_hires
	steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf --cmd "$train_cmd" \
			data/${x}_hires exp/make_hires/$x $mfccdir;
	steps/compute_cmvn_stats.sh data/${x}_hires exp/make_hires/${x} $mfccdir;
	utils/fix_data_dir.sh data/${x}_hires
fi

online_ivector_dir=exp/nnet3/ivectors_swbd_${x}
if [ $ivector -eq 1 ]; then
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    data/${x}_hires $modelDirectory/nnet_online/extractor $online_ivector_dir || exit 1;
fi


decode_dir=$dir/decode_${x}_sw1_tg
if [ $decode -eq 1 ]; then

	steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd" \
        --skip-scoring true \
        --online-ivector-dir $online_ivector_dir \
        $graph_dir data/${x}_hires $decode_dir || exit 1;

fi


if [ $decodeWithoutLM -eq 1 ]; then

	decode_dir=$dir/decode_${x}_without_LM_sw1_tg


    steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd" \
        --skip-scoring true \
        --acwt 100 \
        --beam 150 \
        --online-ivector-dir $online_ivector_dir \
        $graph_dir data/${x}_hires $decode_dir || exit 1;

fi



if [ $phones -eq 1 ]; then
	for i in $(seq $nj) ; do
		lattice-1best ark:$decode_dir/lat.$i ark:- | \
			nbest-to-linear ark:- ark:- | \
			ali-to-phones $modelDirectory/tri4/final.mdl ark:- ark,t:$decode_dir/f/$i.txt
	done


	for n in $(seq $nj); do  cat $decode_dir/f/$n.txt; done > $decode_dir/f/phones.txt


fi