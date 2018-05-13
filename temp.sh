. ./cmd.sh
. ./path.sh

train_set=cv_dev_nz
mfccdir=exp/mfcc
nj=12

utils/fix_data_dir.sh data/${train_set}
steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_16k.conf \
    --cmd "$train_cmd" data/${train_set} exp/make_mfcc/${train_set} $mfccdir;
steps/compute_cmvn_stats.sh data/${train_set} exp/make_mfcc/${train_set} $mfccdir;

utils/fix_data_dir.sh data/${train_set}
echo "============================================="
echo "STANDARD MFCCs DONE!!!"
echo "============================================="