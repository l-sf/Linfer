#pragma once

#include "STrack.h"

struct Object
{
	// x, y, width, height
    float rect[4];
    int label;
    float prob;
};

class BYTETracker
{
public:
	BYTETracker();
	~BYTETracker();

	vector<STrack> update(const vector<Object>& objects);
	tuple<uint8_t, uint8_t, uint8_t> get_color(int idx);
	byte_kalman::Config& config();

private:
	vector<STrack*> joint_stracks(vector<STrack*> &tlista, vector<STrack> &tlistb);
	vector<STrack> joint_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);

	vector<STrack> sub_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);
	void remove_duplicate_stracks(vector<STrack> &resa, vector<STrack> &resb, vector<STrack> &stracksa, vector<STrack> &stracksb);

	void linear_assignment(vector<vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
		vector<vector<int> > &matches, vector<int> &unmatched_a, vector<int> &unmatched_b);
	vector<vector<float> > iou_distance(vector<STrack*> &atracks, vector<STrack> &btracks, int &dist_size, int &dist_size_size);
	vector<vector<float> > iou_distance(vector<STrack> &atracks, vector<STrack> &btracks);
	vector<vector<float> > ious(vector<vector<float> > &atlbrs, vector<vector<float> > &btlbrs);

	double lapjv(const vector<vector<float> > &cost, vector<int> &rowsol, vector<int> &colsol, 
		bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);

private:
	int frame_id;

	vector<STrack> tracked_stracks;
	vector<STrack> lost_stracks;
	vector<STrack> removed_stracks;
	byte_kalman::KalmanFilter kalman_filter;
	byte_kalman::Config& _config = kalman_filter.config();
};