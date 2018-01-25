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


static void pool_triplet_feature_random(
	vector<vector<float>> ground,
	int start, int end, int dim,
	vector<float>& res){
	int ground_num = ground.size();
	if (start > ground.size())
		start = ground.size();
	if (end > ground.size())
		end = ground.size();
	if (start >= end && start - 1 >= 0)
		start -= 1;
	if (start >= end)
		end += 1;
	if (end > ground.size())
		end = ground.size();
	int len = floor((end - start) / 3), lens;
	if (len == 0) lens = 1;
	else lens = len;
	int head_idx = ((rand() % lens) + start);
	int body_idx = (rand() % (lens) + start + len);
	int tail_idx = ((rand() % (lens)) + start + 2 * len);
	for (int j = 0; j < dim; ++j){
		res[j] = 0;
		res[j] += (ground[head_idx][j]);
		res[j] += (ground[body_idx][j]);
		res[j] += (ground[tail_idx][j]);
	}
	for (int i = 0; i < dim; ++i) res[i] /= 3;
}


static void split_fg_bg(
	vector<int>& label,
	vector<pair<int, int>>& fg,
	vector<pair<int, int>>& bg){
	bool end = false;
	int last_label = label[0];
	pair<int, int> temp;
	temp.first = 0;
	for (int i = 0; i < label.size() - 1; ++i){
		if (end){
			temp.first = i;
			end = false;
			last_label = label[i];
			continue;
		}
		if (!end && label[i+1] != last_label){
			temp.second = i;
			end = true;
			if (last_label == 1)
				fg.push_back(temp);
			else
				bg.push_back(temp);
		}
		last_label = label[i];
	}
	if (!end){
		temp.second = label.size() - 1;
		if (last_label == 1)
			fg.push_back(temp);
		else
			bg.push_back(temp);
	}
}

static void get_remove_interval(
	string remove_file,
	map<string, vector<pair<int, int>>>& video_remove){
	std::ifstream infile(remove_file);
	string temp;
	while (std::getline(infile, temp)){
		vector<string> field;
		boost::split(field, temp, boost::is_any_of("\t"));
		string vname = field[0];
		int start_frame = std::stoi(field[1]);
		int end_frame = std::stoi(field[2]);
		pair<int, int> p;
		p.first = start_frame;
		p.second = end_frame;
		video_remove[vname].push_back(p);
	}
}

static bool judge_inside(
	vector<pair<int,int>> interval,
	int pos){
	bool inside = false;
	for (int i = 0; i < interval.size(); ++i){
		if (pos >= interval[i].first && pos <= interval[i].second){
			inside = true;
			break;
		}
	}
	return inside;
}

static void add_tsn_fea(
	int sample_num,
	int bin_fea_c,
	vector<pair<int, int>>gt,
	vector<vector<float>> video_fea,
	int video_label,
	vector<int>& label, 
	vector<vector<float>>& fea,
	int& num){
	vector<float> temp(bin_fea_c, 0);
	for (int ind = 0; ind < sample_num; ++ind){
		int id = rand() % (gt.size());
		int start = gt[id].first;
		int end = gt[id].second;
		pool_triplet_feature_random(video_fea, start, end, bin_fea_c, temp);
		fea.push_back(temp);
		label.push_back(video_label);
		++num;
	}
}

static void add_original_fea(
	int sample_num,
	vector<pair<int, int>>gt,
	vector<vector<float>> video_fea,
	int video_label,
	vector<int>& label,
	vector<vector<float>>& fea,
	int& num,
	bool remove_flag,
	vector<pair<int,int>> remove_frame){
	vector<int> pos;
	for (int i = 0; i < gt.size(); ++i){
		for (int k = gt[i].first; k < gt[i].second; ++k)
			pos.push_back(k);
	}
	float stride = ceil(pos.size()*1.0 / sample_num);
	if (pos.size() != 0 && stride == 0) stride = 1;
	//if (video_label == 1 && remove_flag) stride = 1;
	for (int ind = 0; ind < pos.size() && stride > 0; ind += stride){
		if (remove_flag){
			if (judge_inside(remove_frame, pos[ind])) continue;
		}
		fea.push_back(video_fea[pos[ind]]);
		label.push_back(video_label);
		++num;
	}
}

static void store_write_fea(
	int tsn_flag,
	int bin_fea_c,
	int pos_per_video, int neg_per_video,
	int& write_pos, int& write_neg,
	vector<int>& label, vector<vector<float>>& video_fea,
	vector<int>& write_label,vector<vector<float>>& video_write_fea,
	bool remove_flag, vector<pair<int,int>> remove_proposal){
	// split the foreground and background
	vector<pair<int, int>> fg_gt;
	vector<pair<int, int>> bg_gt;
	split_fg_bg(label, fg_gt, bg_gt);
	// sample fg and bg
	if (fg_gt.size() > 0){
		if (tsn_flag == 1)
			add_tsn_fea(pos_per_video, bin_fea_c, fg_gt, video_fea,
			1, write_label, video_write_fea, write_pos);
		else{
			add_original_fea(pos_per_video, fg_gt, video_fea,
				1, write_label, video_write_fea, write_pos,remove_flag,remove_proposal);
		}
	}
	if (bg_gt.size() > 0){
		if (tsn_flag == 1)
			add_tsn_fea(neg_per_video, bin_fea_c, bg_gt, video_fea,
			0, write_label, video_write_fea, write_neg);
		else{
			add_original_fea(neg_per_video, bg_gt, video_fea,
				0, write_label, video_write_fea, write_neg, remove_flag, remove_proposal);
		}
	}
	// clear all the vector
	for (int free_id = 0; free_id < video_fea.size(); ++free_id)
		video_fea[free_id].clear();
	video_fea.clear();
	label.clear();
}

static void commit_to_datum(
	int bin_fea_c,
	vector<vector<float>>& video_store_fea,
	vector<int>& video_store_label,
	string value,
	scoped_ptr<db::Transaction>& txn,
	const int kMaxKeyLength,
	char* key,
	int& write_count,
	int record_write_video,
	int write_pos,
	int write_neg,
	int shuffle_flag
	){ 
	vector<int> v_index;
	for (int vid = 0; vid < video_store_fea.size(); ++vid) v_index.push_back(vid);
	if (shuffle_flag != 24321)
		shuffle(v_index.begin(), v_index.end());
	for (int id = 0; id < video_store_fea.size(); ++id){
		int ind = v_index[id];
		caffe::Datum datum;
		datum.set_channels(bin_fea_c);
		datum.set_height(1);
		datum.set_width(1);
		for (int k = 0; k < bin_fea_c; ++k){
			datum.add_float_data(video_store_fea[ind][k]);
		}
		datum.set_label(video_store_label[ind]);
		datum.SerializePartialToString(&value);
		int length = _snprintf(key, kMaxKeyLength, "%08d", write_count);
		txn->Put(std::string(key, length), value);
		++write_count;
	}
	txn->Commit();
	LOG(ERROR) << "Have shuffled the stored video number: " << record_write_video << " pos frame: " << write_pos
		<< " neg frame: " << write_neg << " Feature num: " << video_store_fea.size() << std::endl;
	for (int vid = 0; vid < video_store_fea.size(); ++vid) video_store_fea[vid].clear();
	video_store_fea.clear();
	video_store_label.clear();
}

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
		"    convert_frame_fea_point.exe [FLAGS] VIDEO_NAME_LIST BIN_DIR POS_PER_VIDEO NEG_PER_VIDEO\n"
		"	 DB_OUTPUT START_FILE_NUM END_FILE_NUM LABEL_DIR PREFIX SUBFIX Dim SHUFFLE_THRES(!=24321) TSN_FLAG(1: USE tsn) REMOVE_FRAME\n"
		"\n"
		);
	gflags::ParseCommandLineFlags(&args, &argv, true);

	if (args < 9){
		gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/sample_video_feature_with_label");
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


	// number per video
	const int pos_per_video = atoi(argv[3]);
	const int neg_per_video = atoi(argv[4]);

	// set the buff size of each lvdb
	scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
	db->Open(argv[5], db::NEW);
	scoped_ptr<db::Transaction> txn(db->NewTransaction());


	// start file index and end file index
	int start_id = atoi(argv[6]);
	int end_id = atoi(argv[7]);

	// label dir
	string label_dir = argv[8];
	
	// prefix and subfix
	string prefix = argv[9];
	string subfix = argv[10];

	// set the parameter of db
	const int kMaxKeyLength = 100;
	char key[kMaxKeyLength];
	std::string value;

	// read the header for the binary file
	int bin_fea_format;
	int bin_fea_c = atoi(argv[11]);

	// shuffle param
	int shuffle_thres = atoi(argv[12]);

	// tsn sample rule flag
	int tsn_flag = atoi(argv[13]);

	// record the remove frames
	bool remove_flag = false;
	map<string, vector<pair<int, int>>> remove_proposal;

	if (args == 15){
		LOG(ERROR) << "Remove the pos frames for training.";
		get_remove_interval(argv[14], remove_proposal);
		remove_flag = true;
	}

	// record the history fea
	string hitory_video_name;
	bool include_flag = false;
	vector<vector<float>> video_store_fea;
	vector<vector<float>> video_fea;
  	vector<int> video_label;
	vector<int> video_store_label;
	int record_write_video = 0;
	int write_count = 0;
	int write_pos = 0;
	int write_neg = 0;

	// Write datum
	for (int id = start_id; id <= end_id; ++id){
		// input feature and input video name
		string name_temp;
		string feature_dir = bin_dir + "\\" + prefix  +"_" + std::to_string(id) + "." + subfix;
		string name_dir = label_dir + "\\list_" + std::to_string(id) + ".txt";
		std::ifstream feature_in(feature_dir, ios::binary);
		std::ifstream name_in(name_dir, ios::in);
		float* feature = new float[bin_fea_c];
		while (std::getline(name_in, name_temp)){
			vector<string> field;
			boost::split(field, name_temp, boost::is_any_of(" "));
			if (field[0] != hitory_video_name && include_flag){
				vector<pair<int, int>> remove_p;
				if (remove_flag) remove_p = remove_proposal[hitory_video_name];
				// store the video 
				store_write_fea(tsn_flag,bin_fea_c, pos_per_video, neg_per_video, write_pos, write_neg,
					video_label, video_fea, video_store_label, video_store_fea,remove_flag,remove_p);
				include_flag = false;
				++record_write_video;
				// write the video feature
				if (video_store_fea.size() > shuffle_thres)
					commit_to_datum(bin_fea_c, video_store_fea, video_store_label, value, txn, kMaxKeyLength, key,
					write_count, record_write_video, write_pos, write_neg, shuffle_thres);
			}
			// store the feature for one video 
			// process the video one by one
			feature_in.read((char*)(&bin_fea_format), sizeof(bin_fea_format));
			vector<int> des;
			int dim = 1;
			if (bin_fea_format > 100){
				LOG(ERROR) << "Format Error, File Error";
			}
			for (int des_id = 0; des_id < bin_fea_format; ++des_id){
				int temp;
				feature_in.read((char*)(&temp), sizeof(temp));
				des.push_back(temp);
				dim *= temp;
			}
			feature_in.read((char*)(feature), dim*sizeof(float));
			if (bin_fea_c != dim){
				LOG(ERROR) << "The dim has error: set: " << bin_fea_c << " model: " << dim;
			}
			if (std::find(vname_set.begin(), vname_set.end(), field[0]) != vname_set.end()){
				hitory_video_name = field[0];
				include_flag = true;
				if (std::stoi(field[2]) == 1)
					video_label.push_back(1);
				else
					video_label.push_back(0);
				vector<float> video_f;
				for (int j = 0; j < bin_fea_c; ++j){
					video_f.push_back(feature[j]);
				}
				video_fea.push_back(video_f);
			}
		}
		delete feature;
		feature_in.close();
		name_in.close();
		LOG(ERROR) << "Have read the file index : " << id;
	}
	// write the last patch
	if (video_store_fea.size() < shuffle_thres)
		commit_to_datum(bin_fea_c, video_store_fea, video_store_label, value, txn, kMaxKeyLength, key,
		write_count, record_write_video, write_pos, write_neg,shuffle_thres);
}


