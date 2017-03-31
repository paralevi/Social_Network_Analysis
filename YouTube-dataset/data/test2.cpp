#include <bits/stdc++.h>
using namespace std;

struct node {
	int a,b,w;
};
const int MAXN = 15089;
const double prop = 0.8;

vector <int> G[MAXN];
vector <node> L;

node readLineCSV(ifstream &file) {//从csv中读取一行
	int a,b,w;

	string line;
	stringstream s;
	getline(file,line,',');
	s<<line;
	s>>a;
	s.clear();
	getline(file,line,',');
	s<<line;
	s>>b;
	s.clear();
	getline(file,line);
	s<<line;
	s>>w;
	s.clear();
	return node{a,b,w};
}
int getFileLine(ifstream &file) {
	int cnt = 0;
	string tmp;
	while(!file.eof()) {
		getline(file,tmp);
		cnt++;
	}
	file.clear(ios::goodbit);
	file.seekg(ios::beg);
	return cnt;
}
void test() {
	ifstream file("1-edges.csv");
	node t = readLineCSV(file);
	int cnt = getFileLine(file);
	t = readLineCSV(file);
	exit(0);
}

double calSC(node &t) {
		int a_b=0;
		for (auto i:G[t.a])
			for (auto j:G[t.b])
				if (i==j)
					a_b++;
		double t_sc = a_b*1.0/sqrt(G[t.a].size()*G[t.b].size());
		return t_sc;
}

int calCN(node &t) {
	int a_b=0;
	for (auto i:G[t.a])
		for (auto j:G[t.b])
			if (i==j)
				a_b++;
	return a_b;
}

void CommonNeighbor(ifstream &file1, int train_cnt, int test_cnt) {
	vector<int>mean;
	int cn = 0;
	for (int i = 0; i < train_cnt; i++) {
		node &t = L[i];
		int t_cn = calCN(t);
		mean.push_back(t_cn);
	}
	sort(mean.begin(),mean.end());
	cn = mean[mean.size()/2];
	cout << "Common Neighbour"<<endl <<"mean of CN:"<<cn<<endl;
	int prec_cnt = 0;
	for (int i = train_cnt; i < L.size(); i++) {
		node &t=L[i];
		double t_cn = calCN(t);
		if (t_cn >= cn) 
			prec_cnt++;	
	}
	cout << "precision:"<<prec_cnt*1.0/(train_cnt+test_cnt) << endl;
}

void SaltonCosine(ifstream &file1, int train_cnt, int test_cnt) {
	vector <double> mean;
	double sc = 0;
	for (int i = 0; i < train_cnt; i++) {
		node t = readLineCSV(file1);
		double t_sc = calSC(t);
		mean.push_back(t_sc);
	}
	sort(mean.begin(),mean.end());
	sc = mean[mean.size()/2];
	cout << "Salton Consine Similarity"<<endl <<"mean of SC:"<<sc << endl;
	int prec_cnt = 0;
	for (int i = train_cnt; i < L.size(); i++) {
		node &t=L[i];
		double t_sc = calSC(t);
		if (t_sc >= sc) 
			prec_cnt++;
	}
	cout << "precision:"<<prec_cnt*1.0/(train_cnt+test_cnt) << endl;
}

int main() {
	//test();
	ifstream file("out.csv");
	while (!file.eof()) {
		node t = readLineCSV(file1);
		if (t.w <= 2) {
			L.push_back(t);
			G[t.a].push_back(t.b);
			G[t.b].push_back(t.a);
		}

	}
	int total_cnt = L.size();
	int train_cnt = total_cnt * prop;
	int test_cnt = total_cnt - train_cnt;
	cout <<"train:"<< train_cnt<<" test:"<<test_cnt<<endl;
	SaltonCosine(train_cnt, test_cnt);
	CommonNeighbor(train_cnt,test_cnt);
	return 0;
}

