// nnet3bin/nnet3-get-egs.cc

/*
export LD_LIBRARY_PATH=/usr/lib/atlas-base:/exp/sw/kaldi/src/lib:/exp/sw/kaldi/tools/openfst-1.6.2/src/lib/.libs:$LD_LIBRARY_PATH

Compile on Linux
g++ --std=c++11 -I/exp/sw/kaldi/tools/openfst-1.6.2/src/include -I/exp/sw/kaldi/src -I/exp/sw/kaldi/tools/CLAPACK/ -g -O3 nnet3-get-egs-mod.cc -o ./nnet3-get-egs-mod /exp/sw/kaldi/tools/openfst-1.6.2/src/lib/.libs/libfst.so* /exp/sw/kaldi/src/lib/*.so -ldl -lpthread


When 4 args
./nnet3-get-egs-mod --num-pdfs=9054 --num-accents=16 --accent-file=cv-train-accents --scale=true --scale-factor=1.5 --scale-file=australia_accents_train --frame-subsampling-factor=1 --online-ivectors=scp:exp/nnet3/ivectors_cv_train_nz_hires/ivector_online.scp --online-ivector-period=10 --left-context=16 --right-context=12 --compress=true --num-frames=8 "ark,s,cs:utils/filter_scp.pl --exclude exp/nnet3/tdnn_d_sp/egs/valid_uttlist data/cv_train_nz_hires/split20/1/feats.scp | apply-cmvn --norm-means=false --norm-vars=false --utt2spk=ark:data/cv_train_nz_hires/split20/1/utt2spk scp:data/cv_train_nz_hires/split20/1/cmvn.scp scp:- ark:- |" "ark:gunzip -c /home/abhinav/kaldi/accents/exp/tri4_cv_train_nz_ali/ali.1.gz | ali-to-pdf /home/abhinav/kaldi/accents/exp/tri4_cv_train_nz_ali/final.mdl ark:- ark:- | ali-to-post ark:- ark:- |" scp:data/extras/cv-dev-accents ark:-



When 3 args
./nnet3-get-egs-mod --num-pdfs=9054 --num-accents=16 --accent-file=cv-train-accents --scale=true --scale-factor=1.5 --scale-file=australia_accents_train --frame-subsampling-factor=1 --online-ivectors=scp:exp/nnet3/ivectors_cv_train_nz_hires/ivector_online.scp --online-ivector-period=10 --left-context=16 --right-context=12 --compress=true --num-frames=8 "ark,s,cs:utils/filter_scp.pl --exclude exp/nnet3/tdnn_d_sp/egs/valid_uttlist data/cv_train_nz_hires/split20/1/feats.scp | apply-cmvn --norm-means=false --norm-vars=false --utt2spk=ark:data/cv_train_nz_hires/split20/1/utt2spk scp:data/cv_train_nz_hires/split20/1/cmvn.scp scp:- ark:- |" "ark:gunzip -c /home/abhinav/kaldi/accents/exp/tri4_cv_train_nz_ali/ali.1.gz | ali-to-pdf /home/abhinav/kaldi/accents/exp/tri4_cv_train_nz_ali/final.mdl ark:- ark:- | ali-to-post ark:- ark:- |" ark:-




*/




#define HAVE_CLAPACK 1

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <sstream>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "hmm/posterior.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h"
#include "util/kaldi-table.h"

using namespace std;


namespace kaldi {
namespace nnet3 {


static bool ProcessFile(const GeneralMatrix &feats,
                        int32 accent_class,
                        const MatrixBase<BaseFloat> *ivector_feats,
                        int32 ivector_period,
                        const Posterior &pdf_post,
                        const std::string &utt_id,
                        bool compress,
                        int32 num_pdfs,
                        int32 num_accents,
                        unordered_map<string, int32>& uttIdToAccent,
                        bool scale,
                        unordered_set<string>& scale_file_set,
                        float scale_factor,
                        int32 length_tolerance,
                        UtteranceSplitter *utt_splitter,
                        NnetExampleWriter *example_writer) {
  int32 num_input_frames = feats.NumRows();

  int32 accent=-1;
  if(num_accents!=-1)
    accent=uttIdToAccent[utt_id];

  if (!utt_splitter->LengthsMatch(utt_id, num_input_frames,
                                  static_cast<int32>(pdf_post.size()),
                                  length_tolerance))
    return false;  // LengthsMatch() will have printed a warning.

  std::vector<ChunkTimeInfo> chunks;

  utt_splitter->GetChunksForUtterance(num_input_frames, &chunks);

  if (chunks.empty()) {
    KALDI_WARN << "Not producing egs for utterance " << utt_id
               << " because it is too short: "
               << num_input_frames << " frames.";
  }

  // 'frame_subsampling_factor' is not used in any recipes at the time of
  // writing, this is being supported to unify the code with the 'chain' recipes
  // and in case we need it for some reason in future.
  int32 frame_subsampling_factor =
      utt_splitter->Config().frame_subsampling_factor;

  for (size_t c = 0; c < chunks.size(); c++) {
    const ChunkTimeInfo &chunk = chunks[c];

    int32 tot_input_frames = chunk.left_context + chunk.num_frames +
        chunk.right_context;

    int32 start_frame = chunk.first_frame - chunk.left_context;

    GeneralMatrix input_frames;
    ExtractRowRangeWithPadding(feats, start_frame, tot_input_frames,
                               &input_frames);

    // 'input_frames' now stores the relevant rows (maybe with padding) from the
    // original Matrix or (more likely) CompressedMatrix.  If a CompressedMatrix,
    // it does this without un-compressing and re-compressing, so there is no loss
    // of accuracy.

    NnetExample eg;
    // call the regular input "input".
    eg.io.push_back(NnetIo("input", -chunk.left_context, input_frames));

    if (ivector_feats != NULL) {
      // if applicable, add the iVector feature.
      // choose iVector from a random frame in the chunk
      int32 ivector_frame = RandInt(start_frame,
                                    start_frame + num_input_frames - 1),
          ivector_frame_subsampled = ivector_frame / ivector_period;
      if (ivector_frame_subsampled < 0)
        ivector_frame_subsampled = 0;
      if (ivector_frame_subsampled >= ivector_feats->NumRows())
        ivector_frame_subsampled = ivector_feats->NumRows() - 1;
      Matrix<BaseFloat> ivector(1, ivector_feats->NumCols());
      ivector.Row(0).CopyFromVec(ivector_feats->Row(ivector_frame_subsampled));
      eg.io.push_back(NnetIo("ivector", 0, ivector));
    }

    // Note: chunk.first_frame and chunk.num_frames will both be
    // multiples of frame_subsampling_factor.
    int32 start_frame_subsampled = chunk.first_frame / frame_subsampling_factor,
        num_frames_subsampled = chunk.num_frames / frame_subsampling_factor;

    Posterior labels(num_frames_subsampled);
    Posterior labels_accents(num_frames_subsampled);

    // TODO: it may be that using these weights is not actually helpful (with
    // chain training, it was not), and that setting them all to 1 is better.
    // We could add a boolean option to this program to control that; but I
    // don't want to add such an option if experiments show that it is not
    // helpful.
    for (int32 i = 0; i < num_frames_subsampled; i++) {
      int32 t = i + start_frame_subsampled;
      if (t < pdf_post.size())
      {
        labels[i] = pdf_post[t];
        if(num_accents!=-1)
        {
          vector<pair<int32, BaseFloat > > temp;
          pair<int32, BaseFloat > t1=make_pair(accent, 1.0);
          temp.push_back(t1);  
          labels_accents[i]=temp;
        }
        
      }
      for (std::vector<std::pair<int32, BaseFloat> >::iterator
               iter = labels[i].begin(); iter != labels[i].end(); ++iter)
        iter->second *= chunk.output_weights[i];
    }
    

    if(scale && scale_file_set.find(utt_id)!=scale_file_set.end())
    {
      KALDI_WARN << utt_id << "exists, Scaling";
      ScalePosterior(scale_factor, &labels);
/*
      for (int32 i = 0; i < num_frames_subsampled; i++) {
        KALDI_WARN << "size:" << labels[i].size();
        getchar();
      for (std::vector<std::pair<int32, BaseFloat> >::iterator
               iter = labels_accents[i].begin(); iter != labels_accents[i].end(); ++iter)
        KALDI_WARN << iter->first;
    }*/
    
    }

    eg.io.push_back(NnetIo("output", num_pdfs, 0, labels));
    if(num_accents!=-1)
      eg.io.push_back(NnetIo("output-accent", num_accents, 0, labels_accents));

    if (compress)
      eg.Compress();

    std::ostringstream os;
    os << utt_id << "-" << chunk.first_frame;

    std::string key = os.str(); // key is <utt_id>-<frame_id>
    //KALDI_WARN << key;
    example_writer->Write(key, eg);
  }
  return true;
}

} // namespace nnet3
} // namespace kaldi


void ReadScaleFile(const char* scale_file, unordered_set<string>& scale_file_set)
{ 
  string line;
  ifstream myFile(scale_file);
  if(myFile.is_open())
  {
    while(getline(myFile, line))
    {
      scale_file_set.insert(line);
    }
  }

}

void ReadAccentsFile(const char* accent_file, unordered_map<string, int32>& uttIdToAccent)
{ 
  string line;
  ifstream myFile(accent_file);
  if(myFile.is_open())
  {
    while(getline(myFile, line))
    {
      string uttId;
      string delimiter=" ";
      int pos;
      while((pos=line.find(delimiter))!=string::npos)
      {
        uttId=line.substr(0,pos);
        line.erase(0,pos+delimiter.length());
      }
      int32 accent=stoi(line);
      uttIdToAccent.insert(make_pair(uttId, accent));
    }
  }

}




int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Get frame-by-frame examples of data for nnet3 neural network training.\n"
        "Essentially this is a format change from features and posteriors\n"
        "into a special frame-by-frame format.  This program handles the\n"
        "common case where you have some input features, possibly some\n"
        "iVectors, and one set of labels.  If people in future want to\n"
        "do different things they may have to extend this program or create\n"
        "different versions of it for different tasks (the egs format is quite\n"
        "general)\n"
        "\n"
        "Usage:  nnet3-get-egs [options] <features-rspecifier> "
        "<pdf-post-rspecifier> <egs-out>\n"
        "\n"
        "An example [where $feats expands to the actual features]:\n"
        "nnet3-get-egs --num-pdfs=2658 --num-accents=10 --accent-file=all_accents --left-context=12 --right-context=9 --num-frames=8 \"$feats\"\\\n"
        "\"ark:gunzip -c exp/nnet/ali.1.gz | ali-to-pdf exp/nnet/1.nnet ark:- ark:- | ali-to-post ark:- ark:- |\" \\\n"
        "   ark:- \n"
        "See also: nnet3-chain-get-egs, nnet3-get-egs-simple\n";


    bool compress = true, scale=false;
    float scale_factor=-1;
    string scale_file="";
    string accents_file="";
    int32 num_pdfs = -1, num_accents=-1, length_tolerance = 100,
        targets_length_tolerance = 2,  
        online_ivector_period = 1;

    unordered_set<string> scale_file_set;
    unordered_map<string, int32> uttIdToAccent;
        

    ExampleGenerationConfig eg_config;  // controls num-frames,
                                        // left/right-context, etc.

    std::string online_ivector_rspecifier;

    ParseOptions po(usage);

    po.Register("num-accents", &num_accents, "The number of accents for adding the multi-task output"
                  "layer for learning accents");
    po.Register("accents-file", &accents_file, "The path to file which contains accents for each utterance"
                  "keyed by utterance ids");

    po.Register("scale", &scale, "If true, some of the posteriors are mentioned by the "
                  "in the scale-file are scaled by the given scale-factor");
    po.Register("scale-factor", &scale_factor, "If scale is true, the posteriors for the utterances"
                  "in the scale-file are scaled by this value.");
    po.Register("scale-file", &scale_file, "If scale is true, the posteriors of utterances given in"
                  "are scaled by the scale-factor.");
    po.Register("compress", &compress, "If true, write egs with input features "
                "in compressed format (recommended).  This is "
                "only relevant if the features being read are un-compressed; "
                "if already compressed, we keep we same compressed format when "
                "dumping egs.");
    po.Register("num-pdfs", &num_pdfs, "Number of pdfs in the acoustic "
                "model");
    po.Register("num-accents", &num_accents, "Number of pdfs in the acoustic "
                "model");
    po.Register("ivectors", &online_ivector_rspecifier, "Alias for "
                "--online-ivectors option, for back compatibility");
    po.Register("online-ivectors", &online_ivector_rspecifier, "Rspecifier of "
                "ivector features, as a matrix.");
    po.Register("online-ivector-period", &online_ivector_period, "Number of "
                "frames between iVectors in matrices supplied to the "
                "--online-ivectors option");
    po.Register("length-tolerance", &length_tolerance, "Tolerance for "
                "difference in num-frames between feat and ivector matrices");
    po.Register("targets-length-tolerance", &targets_length_tolerance, 
                "Tolerance for "
                "difference in num-frames (after subsampling) between "
                "feature matrix and posterior");
    eg_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    if(scale)
    {
      if(scale_factor==-1 || scale_file=="")
      {
        KALDI_ERR << "--scale-factor and --scale-file are required if scale is true";
        exit(1);
      }
    }


    if(scale)
    {
      const char* scale_file_str=scale_file.c_str();
      ReadScaleFile(scale_file_str, scale_file_set);
    }


    if(num_accents!=-1)
    {
      if(accents_file!="")
      {
        const char* accents_file_str=accents_file.c_str();
        ReadAccentsFile(accents_file_str, uttIdToAccent);
      }
      else
      {
        KALDI_ERR << "--accents_file is empty even though num-accents is given, something is wrong!";
        exit(1);
      }
    }


    if (num_pdfs <= 0)
      KALDI_ERR << "--num-pdfs options is required.";

    eg_config.ComputeDerived();
    UtteranceSplitter utt_splitter(eg_config);

    std::string feature_rspecifier = po.GetArg(1),
        pdf_post_rspecifier = po.GetArg(2),
        //accent_class_rspecifier = po.GetArg(3), //added
        examples_wspecifier = po.GetArg(3);

    // SequentialGeneralMatrixReader can read either a Matrix or
    // CompressedMatrix (or SparseMatrix, but not as relevant here),
    // and it retains the type.  This way, we can generate parts of
    // the feature matrices without uncompressing and re-compressing.
    SequentialGeneralMatrixReader feat_reader(feature_rspecifier);
    RandomAccessPosteriorReader pdf_post_reader(pdf_post_rspecifier);
    NnetExampleWriter example_writer(examples_wspecifier);
    RandomAccessBaseFloatMatrixReader online_ivector_reader(
        online_ivector_rspecifier);
    //RandomAccessInt32Reader accent_class_reader(accent_class_rspecifier);              //added

    int32 num_err = 0;

    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const GeneralMatrix &feats = feat_reader.Value();
      //feats.Write()
      //KALDI_WARN << "Dimension: " << feats.NumCols();
      //KALDI_WARN << "Key: " << key;

      //int32 accent_class;
       /*if (!accent_class_reader.HasKey(key))
        {
            KALDI_WARN << "No accent for utterance " << key;
            num_err++;
            continue;
        }
        else
        {
            accent_class=accent_class_reader.Value(key);
            KALDI_WARN << "Accent Found for utterance " << key << ":" << accent_class;
            continue;
        }*/




      if (!pdf_post_reader.HasKey(key)) {
        KALDI_WARN << "No pdf-level posterior for key " << key;
        num_err++;
      } else {
        const Posterior &pdf_post = pdf_post_reader.Value(key);
        const Matrix<BaseFloat> *online_ivector_feats = NULL;
        int32 accent_class;
        if (!online_ivector_rspecifier.empty()) {
          if (!online_ivector_reader.HasKey(key)) {
            KALDI_WARN << "No iVectors for utterance " << key;
            num_err++;
            continue;
          } else {
            // this address will be valid until we call HasKey() or Value()
            // again.
            online_ivector_feats = &(online_ivector_reader.Value(key));
          }
        }

        //added
        /*if (!accent_class_reader.HasKey(key))
        {
            KALDI_WARN << "No iVectors for utterance " << key;
            num_err++;
            continue;
        }
        else
        {
            accent_class=accent_class_reader.Value(key);
        }*/


        if (online_ivector_feats != NULL &&
            (abs(feats.NumRows() - (online_ivector_feats->NumRows() *
                                    online_ivector_period)) > length_tolerance
             || online_ivector_feats->NumRows() == 0)) {
          KALDI_WARN << "Length difference between feats " << feats.NumRows()
                     << " and iVectors " << online_ivector_feats->NumRows()
                     << "exceeds tolerance " << length_tolerance;
          num_err++;
          continue;
        }

        if (!ProcessFile(feats, accent_class, online_ivector_feats, online_ivector_period,
                         pdf_post, key, compress, num_pdfs, num_accents, uttIdToAccent, scale, scale_file_set, scale_factor,
                         targets_length_tolerance,
                         &utt_splitter, &example_writer))
          num_err++;
      }
    }
    if (num_err > 0)
      KALDI_WARN << num_err << " utterances had errors and could "
          "not be processed.";
    // utt_splitter prints stats in its destructor.
    return utt_splitter.ExitStatus();
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
