/*
	Author: Shivam Tayal
			Shrestha Kumar
*/


#include <bits/stdc++.h>
using namespace std;
vector<long double> operator*(const vector<long double> &a, const vector<vector<long double> > &b);
vector<long double> operator*(const vector<vector<long double> >&a, const vector<long double> &b);
vector<vector<long double> > operator*(const vector< vector<long double> > &a, const vector<vector<long double> > &b);
vector<vector<long double> > operator*(const long double &a, const vector<vector<long double> > &b);
vector<vector<long double> > operator+(const vector<vector<long double> > &a, const vector<vector<long double> > &b);
vector<vector<long double> > operator-(const vector<vector<long double> > &a, const vector<vector<long double> > &b);
vector<vector<long double> > operator%(const vector<long double> &a, const vector<long double> &b);
vector<long double> operator*(const vector<long double> &a, const vector<long double> &b);
vector<long double> operator-(const vector<long double> &a, const vector<long double> &b);
ostream& operator<<(ostream& os, const vector<long double> &v);
ostream& operator<<(ostream& os, const vector<vector<long double> > &v);

vector<long double> split(string s,char delimiter);

class NN{
	int layer;
	vector< vector<vector<long double> > > weights;
	long double alpha,beta;
	long double sigmoid(long double x);
	vector<long double> sigmoid(vector<long double> x);
	vector<long double> forwardPass(vector<long double> _input); 
	void backpropagation(vector<long double> _input,vector<long double> _output);
  public:
  	NN(int _layer, int inputSize,vector<int> hiddenSize,int outputSize,long double _alpha,long double _beta);
  	void pickweights(string address);
  	void saveweights(string address);
	void train(vector<long double> _input,vector<long double> _output); 
	int getoutput(vector<long double> _input);
};

int main(){
	string Data;

	ifstream trainData("sign_mnist_train.csv");
	getline(trainData,Data);
	int inputSize = count(Data.begin(), Data.end(), ',')-1;
	int outputSize = 25;
	vector<int> hiddenSize = {10};

	NN NeuralNetwork(3,inputSize,hiddenSize,outputSize,0.1,0.1);
	NeuralNetwork.saveweights("weights1.csv");
	int i = 100,j=20;
	while(trainData.good() and i>0){
		getline(trainData,Data);
		if(Data == "")
			break;
		vector<long double> data = split(Data,',');
		vector<long double> output(outputSize,0);
		output[0] = data[0]; 
		NeuralNetwork.train(vector<long double>(data.begin()+1,data.end()),output);
		i--;
	}
	trainData.close();

	NeuralNetwork.saveweights("weights.csv");
	ifstream testData("sign_mnist_test.csv");
	ofstream results("results.csv");
	getline(testData,Data);
	while(testData.good() and j>0){
		getline(testData,Data);
		if(Data == "")
			break;
		vector<long double> data = split(Data,',');
		results<<NeuralNetwork.getoutput(vector<long double>(data.begin()+1,data.end()))<<','<<data[0]*(long double)25<<'\n';
		j--;
	}
	testData.close();
	results.close();
	cout<<"done\n";
	return 0;
}

long double NN::sigmoid(long double x){
	return 1.0/(1.0 + (long double)exp(-x/100000ll));
}

vector<long double> NN::sigmoid(vector<long double> x){
	for(int i=0;i<x.size();i++){
		x[i] = sigmoid(x[i]);
	}
	return x;
}

NN::NN(int _layer, int inputSize,vector<int> hiddenSize,int outputSize,long double _alpha,long double _beta){
  	layer = _layer;
  	alpha = _alpha;
  	beta = _beta;
  	weights.resize(layer-1);
  	weights[0].resize(inputSize+1,vector<long double> (hiddenSize[0]));
  	for(int i=1;i<layer-2;i++){
  		weights[i].resize(hiddenSize[i-1]+1,vector<long double> (hiddenSize[i]));
  	}
  	weights[layer-2].resize(hiddenSize[layer-3]+1,vector<long double> (outputSize));
  	srand(0);
  	for(int i=0;i<layer-1;i++){
  		for(int j=0;j<weights[i].size();j++){
  			for(int k=0;k<weights[i][j].size();k++){
  				weights[i][j][k] = rand()/((long double)RAND_MAX);
  			}
  		}
  	}

}

void NN::pickweights(string address){
  	ifstream file (address);
  	string data;
  	getline(file,data);
  	for(int i=0;i<layer-1;i++){
  		int k=0;
  		while(file.good()){
  			getline(file,data);
  			if(data.find("Layer") != string::npos)
  				break;
  			weights[i][k++] = split(data,',');
  		}
  	}
  	file.close();
}

void NN::saveweights(string address){
  	ofstream file (address);
  	for(int i=0;i<layer-1;i++){
  		file<<"Layer "<<i<<'-'<<i+1<<'\n';
  		file<<weights[i];
  	}
  	file.close();
}

vector<long double> NN::forwardPass(vector<long double> _input){
	vector<long double> _output;
	for(int i=0;i<layer-1;i++){
		_input.push_back(1.0);	
		_output = sigmoid(_input*weights[i]);	
		_input = _output;
	}
	for(int i=0;i<25;i++)
		_output[i] = rand()/((long double)RAND_MAX);
	
	return _output;
}

void NN::backpropagation(vector<long double> _input,vector<long double> _output){
	vector<vector<long double> > output(layer);
	for(int i=0;i<layer-1;i++){
		output[i] = _input;
		_input.push_back(1.0);
		_input = sigmoid(_input*weights[i]);
	}

	output[layer-1] = _input;

	vector<long double> one;
	vector<long double> delta(weights[layer-2][0].size());		
	one.resize(output[layer-1].size(),1.0);
	delta = (output[layer-1]*(one-output[layer-1]))*(output[layer-1]-_output);
	weights[layer-2] = (1+beta)*weights[layer-2] - (alpha*(output[layer-2]%delta));
	for(int i=layer-3;i>=0;i--){
		one.resize(output[i+1].size(),1.0);
		delta = (output[i+1]*(one-output[i+1]))*(weights[i]*delta);
		weights[i] = (1+beta)*weights[i] - alpha*(output[i]%delta);
	}
	
	for(int i=0;i<layer-1;i++)
		for(int j=0;j<weights[i].size();j++)
			weights[i][j] = sigmoid(weights[i][j]);
}

void NN::train(vector<long double> _input,vector<long double> _output){
	backpropagation(_input,_output);
}

int NN::getoutput(vector<long double> _input){
	vector<long double> _output = forwardPass(_input);
	return (int)(_output[0]*(long double)25);
}

vector<long double> split(string s,char delimiter){
	vector<long double> res;
	string t = "";
	size_t sz;
	for(int i=0;i<s.length();i++){
		if(s[i] == delimiter){
			if(t != ""){
				if(res.empty())
					res.push_back(stod(t,&sz)/(long double)25);
				else	
					res.push_back(stod(t,&sz)/(long double)255);
			}
			t = "";
		}
		else{
			t += s[i];
		}
	}
	return res;
}

vector<long double> operator*(const vector<vector<long double> >&a, const vector<long double> &b){
	vector<long double> c(a.size()-1,0);
	for(int j=0;j<a.size()-1;j++){
		for(int k=0;k<a[j].size();k++){
			c[j] += a[j][k]*b[k];
		}
	}
    return c;
}

vector<long double> operator*(const vector<long double> &a, const vector<vector<long double> > &b){
	vector<long double> c(b[0].size(),0);
	for(int j=0;j<b[0].size();j++){
		for(int k=0;k<b.size();k++){
			c[j] += a[k]*b[k][j];
		}
	}
    return c;
}

vector<vector<long double> > operator*(const vector< vector<long double> > &a, const vector<vector<long double> > &b){
	vector<vector<long double> > c(a.size(),vector<long double> (b[0].size()));
	for(int i=0;i<a.size();i++){
		for(int j=0;j<b[i].size();j++){
			for(int k=0;k<a[i].size();k++){
				c[i][j] += a[i][k]*b[k][j];
			}
		}
	}
    return c;
}

vector<vector<long double> > operator*(const long double &a, const vector<vector<long double> > &b){
	vector<vector<long double> > c = b;
	for(int i=0;i<b.size();i++){
		for(int j=0;j<b[i].size();j++){
			c[i][j] = a*b[i][j];
		}
	}
    return c;
}

vector<vector<long double> > operator+(const vector<vector<long double> > &a, const vector<vector<long double> > &b){
	vector<vector<long double> > c = a;
	for(int i=0;i<a.size();i++){
		for(int j=0;j<a[i].size();j++){
			c[i][j] += b[i][j];
		}
	}
    return c;
}

vector<vector<long double> > operator-(const vector<vector<long double> > &a, const vector<vector<long double> > &b){
	vector<vector<long double> > c = a;
	for(int i=0;i<a.size();i++){
		for(int j=0;j<a[i].size();j++){
			c[i][j] -= b[i][j];
		}
	}
    return c;
}

vector<vector<long double> > operator%(const vector<long double> &a, const vector<long double> &b){
	vector<vector<long double> > c(a.size(),vector<long double>(b.size()));
	for(int i=0;i<a.size();i++){
		for(int j=0;j<b.size();j++){
			c[i][j] = a[i]*b[j];
		}
	}
	c.push_back(b);
    return c;
}

vector<long double> operator*(const vector<long double> &a, const vector<long double> &b){
	vector<long double> c(a.size(),0);
	for(int i=0;i<a.size();i++){
		c[i] = a[i]*b[i];
	}
    return c;
}

vector<long double> operator-(const vector<long double> &a, const vector<long double> &b){
	vector<long double> c(a.size(),0);
	for(int i=0;i<a.size();i++){
		c[i] = a[i]-b[i];
	}
    return c;
}

ostream& operator<<(ostream& os, const vector<long double> &v){
    for (int i=0;i<v.size()-1;i++){
    	os<<v[i]<<',';
    }
    os<<v[v.size()-1]<<'\n';
    return os;
}

ostream& operator<<(ostream& os, const vector<vector<long double> > &v){
    for (int i=0;i<v.size();i++){
    	for(int j=0;j<v[i].size()-1;j++){
        	os<<v[i][j]<<',';
        }
        os<<v[i][v[0].size()-1]<<'\n';
    }
    return os;
}