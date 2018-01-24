// For the video proposal research

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "boost/scoped_ptr.hpp"
#include "stdint.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"
#include "boost/algorithm/string.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;
using std::string;
using std::vector;

DEFINE_string(backend, "lmdb",
	"The backend {lmdb, leveldb} for storing the result");

int main(int args, char** argv){
	argv[0] = "D:/log";
	::google::InitGoogleLogging(argv[0]);
	FLAGS_alsologtostderr = 1;
#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::SetUsageMessage("sample feature to leveldb/lmdb\n"
		"for the hashing/feature learning.\n"
		"Usage:\n"
		"    sample_video_feature_with_pair.exe [FLAGS] VIDEO_LABEL_NAME_LIST BIN_DIR PAIR_EACH_VIDEO DB_OUTPUT START_FILE_NUM END_FILE_NUM LABEL_DIR PREFIX SUBFIX DIM\n"
		"\n"
		);
	gflags::ParseCommandLineFlags(&args, &argv, true);

	if (args < 7){
		gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/sample_video_feature_label");
		return 1;
	}

	// read the Video List
	std::ifstream vname_in(argv[1], ios::in);
	std::vector<string> vname_set;
	string name;
	while (std::getline(vname_in, name)){
		vname_set.push_back(name);
	}
	// bin file diretory
	string bin_dir = argv[2];

	// pair number each video
	const int pair_per_video = atoi(argv[3]);

	// set the buff size of each lvdb
	scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
	db->Open(argv[4], db::NEW);
	scoped_ptr<db::Transaction> txn(db->NewTransaction());


	// start file index and end file index
	int start_id = atoi(argv[5]);
	int end_id = atoi(argv[6]);

	// label dir
	string label_dir = argv[7];

	// prefix and sub fix
	string prefix = argv[8];
	string subfix = argv[9];

	// set the parameter of db
	const int kMaxKeyLength = 100;
	char key[kMaxKeyLength];
	std::string value;

	// read the header for the binary file
	int bin_fea_format;
	int bin_fea_c = atoi(argv[10]);

	// record the history fea
	string hitory_video_name;
	bool include_flag = false;
	vector<vector<float>> video_pos_fea;
	vector<vector<float>> video_neg_fea;
	vector<float> video_label;
	int record_write_video = 0;
	int write_count = 0;
	int write_pos = 0;
	int write_neg = 0;

	// Write datum
	for (int id = start_id; id <= end_id; ++id){
		// input feature and input video name
		string name_temp;
		string feature_dir = bin_dir + "\\" + prefix + "_" + std::to_string(id) + "." + subfix;
		string name_dir = label_dir + "\\list_" + std::to_string(id) + ".txt";
		std::ifstream feature_in(feature_dir, ios::binary);
		std::ifstream name_in(name_dir, ios::in);
		float* feature = new float[bin_fea_c];
		while (std::getline(name_in, name_temp)){
			vector<string> field;
			boost::split(field, name_temp, boost::is_any_of(" "));
			// sample the pos and neg
			if (field[0] != hitory_video_name && include_flag){
				int pos_num = video_pos_fea.size();
				int neg_num = video_neg_fea.size();
				// sample pair from the pos and neg_num
				if (neg_num != 0 && pos_num != 0){
					for (int i = 0; i < pair_per_video; ++i){
						caffe::Datum datum;
						datum.set_channels(bin_fea_c * 2);
						datum.set_height(1);
						datum.set_width(1);
						int p_position = (rand() % pos_num);
						int n_position = (rand() % neg_num);
						// write pos
						for (int k = 0; k < bin_fea_c; ++k)
							datum.add_float_data(video_pos_fea[p_position][k]);
						// write neg
						for (int k = 0; k < bin_fea_c; ++k)
							datum.add_float_data(video_neg_fea[n_position][k]);
						datum.set_label(0);
						datum.SerializePartialToString(&value);
						int length = _snprintf(key, kMaxKeyLength, "%08d", write_count);
						txn->Put(std::string(key, length), value);
						++write_count;
					}
				}
				// clear all the vector
				for (int free_id = 0; free_id < pos_num; ++free_id)
					video_pos_fea[free_id].clear();
				video_pos_fea.clear();
				for (int free_id = 0; free_id < neg_num; ++free_id)
					video_neg_fea[free_id].clear();
				video_neg_fea.clear();
				include_flag = false;
				++record_write_video;
				if (record_write_video % 100 == 0){
					txn->Commit();
					LOG(ERROR) << "Have writen the video: " << record_write_video;
				}
			}

			// store the feature for one video 
			// process the video one by one
			feature_in.read((char*)(&bin_fea_format), sizeof(bin_fea_format));
			vector<int> des;
			int dim = 1;
			for (int des_id = 0; des_id < bin_fea_format; ++des_id){
				int temp;
				feature_in.read((char*)(&temp), sizeof(temp));
				des.push_back(temp);
				dim *= temp;
			}
			if (bin_fea_c != dim){
				LOG(ERROR) << "The dim has error: set: " << bin_fea_c << " model: " << dim;
 			}
			feature_in.read((char*)(feature), dim*sizeof(float));

			if (std::find(vname_set.begin(), vname_set.end(), field[0]) != vname_set.end()){
				hitory_video_name = field[0];
				include_flag = true;
				if (std::stoi(field[2]) == 1){
					vector<float> video_f;
					for (int j = 0; j < bin_fea_c; ++j){
						video_f.push_back(feature[j]);
					}
					video_pos_fea.push_back(video_f);
				}
				if (std::stoi(field[2]) == 0){
					vector<float> video_f;
					for (int j = 0; j < bin_fea_c; ++j){
						video_f.push_back(feature[j]);
					}
					video_neg_fea.push_back(video_f);
				}
			}
		}
		delete feature;
		feature_in.close();
		LOG(ERROR) << "Have read the file index : " << id;
	}

	txn->Commit();
	LOG(ERROR) << "Success sample features.";
}


