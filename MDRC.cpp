#include<iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <string>
#include <string.h>
#include <map>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <set>
#include<utility>  
#include<stack>  
#include <list>
#include <unordered_set>  
#include <queue>
#include<unordered_map>  
#include<bitset> 

#include <tuple>
#include <iomanip>  // for std::setprecision


const int N = 400000+5;
const double beta=0.43845;
const double the_ratio=0.35497; 
const double l=1;
//const double epsilon=0.2;
using namespace std; 

int n, m; /* number of nodes, arcs */

vector<string> username;
map<string, int> user2ID;	// a map from username to userid
map<int ,string> ID2User;   // a map from userid to username

vector<int> InputEdges;		//edges
vector<double> EdgeProb;	//p_{uv}
vector<double> EdgeProb_sn;	//
map<string, vector<double> > NodeProb;	//p_u^o
map<string, double > userCost;	//Cost
map<string, double > userReputation;	//Reputation
map<string, int> in_deg;			//in-degree
map<string, int> foe_num;			//number of foe links
map<string, long double> pa;
map<string, long double> pb;
map<string, double> viewpoint;	//viewpoint of user
vector<string> S;	//seeds
 
vector<list<pair<int,double> > > adj_out;
vector<list<pair<int,double> > > adj_in;
vector<list<pair<int,double> > > adj_out_reverse;
vector<list<pair<int,double> > > adj_sn_reverse;

const int o_len=21;
double opinions[o_len]={-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
//const int o_len=5;

//double opinions[o_len]={-1.0,-0.5,0.0,0.5,1.0};

struct ResultType{
	string name;
	double finaldiver;
	double runningtime;
	int seedsize;
};

double a_rand(){
	   
    double random_number = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
    return random_number;
}

/* obtain social network and set parameters */
void readGraph(const char* file,double gamma){
	//readfile, obtain vertexs edges
	int x,y;
	int count_edges=0;
	int count_vertices=0;
	string user_a,user_b;
	long double prob_edge;
	long double prob_a,prob_b;
	
	
	
	map<string, int>::iterator it;
	ifstream infile(file);
	
	while(infile>>user_a>>user_b>>prob_a>>prob_b) {
		//for a
		it=user2ID.find(user_a);
		if(it!=user2ID.end()){
			x=it->second;
			pa[user_a]+=prob_a;
			pb[user_a]+=prob_b;
		}
		else{
			username.push_back(user_a);
			user2ID[user_a]=count_vertices;
			x=count_vertices++;
			pa[user_a]=prob_a;
			pb[user_a]=prob_b;
		}
		
		//for b
		it=user2ID.find(user_b);
		if(it!=user2ID.end()){
			y=it->second;
			in_deg[user_b]+=1;

		}
		else{
			username.push_back(user_b);
			user2ID[user_b]=count_vertices;
			y=count_vertices++;
			in_deg[user_b]=1;

		}		
		
		
		
		count_edges++;
		InputEdges.push_back(x);
		InputEdges.push_back(y);	
	}
	
	n=count_vertices;
	m=count_edges;
	
	cout<<"number of users:"<<n<<" number of edges:"<<m<<"\n";
	
	//cost
	vector<string>::iterator itForCost;
	for(itForCost=username.begin();itForCost!=username.end();itForCost++){
		userCost[*itForCost]=1.0+gamma*in_deg[*itForCost];
	}

	 
	//viewpoint
	map<string, long double>::iterator it_for_pa;
	map<string, long double>::iterator it_for_pb;
	vector<string>::iterator it_vector;
	
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();  
    std::mt19937 gen(seed);  
  
    //[-1, 1]
    std::uniform_real_distribution<> dis(-1.0, 1.0);  
    
	// [0, 1]
    std::uniform_real_distribution<> dis2(0, 1.0);  
    

	
	for(it_vector=username.begin();it_vector!=username.end();it_vector++)
	{
		double randomNumber = dis(gen);
		viewpoint[*it_vector]=round(randomNumber * 10) / 10;
	}
	//map<string, int>::iterator it_for_indeg;	
	for(it_for_pa=pa.begin(),it_for_pb=pb.begin();it_for_pa!=pa.end();it_for_pa++,it_for_pb++){	
		viewpoint[it_for_pa->first]=(it_for_pa->second*2)/(it_for_pa->second+it_for_pb->second)-1;
		viewpoint[it_for_pa->first]=round(viewpoint[it_for_pa->first] * 10) / 10;
	}
	
	//reputation
	vector<string>::iterator itForRepu;
	for(itForRepu=username.begin();itForRepu!=username.end();itForRepu++){
		double randomNumber = dis2(gen);
		userReputation[*itForRepu]=round(randomNumber * 10) / 10;;
	}

	
	
	//EdgeProb
	ifstream infile2(file);
	while(infile2>>user_a>>user_b>>prob_a>>prob_b){
		EdgeProb.push_back(1.0/in_deg[user_b]*(userReputation[user_a]+0.5) );
		EdgeProb_sn.push_back(1.0/in_deg[user_b]);

	}
	
	//NodeProb
	vector<string>::iterator itForNodeProb;
	for(itForNodeProb=username.begin();itForNodeProb!=username.end();itForNodeProb++){
		for(int l=0;l<21;l++){
			double puo=0.0;
			if(viewpoint[*itForNodeProb]>=-1.0+l*0.1) puo=1-(viewpoint[*itForNodeProb]+1.0-l*0.1)/2.0;
			else puo=1-(-1.0+l*0.1-viewpoint[*itForNodeProb])/2.0;
			
			NodeProb[*itForNodeProb].push_back(puo);
		}
		

	}	
	

	
	
	if(adj_out.size()<n) adj_out.resize(n);
	if(adj_out_reverse.size()<n) adj_out_reverse.resize(n);
	if(adj_sn_reverse.size()<n) adj_sn_reverse.resize(n);
	//Construct adjacency table (out)
	for (int j=0; j<InputEdges.size(); j+=2){
		//edge x -> y
		x = InputEdges.at(j);	 
		y = InputEdges.at(j+1);
		int i=j/2;
		adj_out[x].push_back(pair<int,double>(y,EdgeProb[i]));
		adj_out_reverse[y].push_back(pair<int,double>(x,EdgeProb[i]));
		adj_sn_reverse[y].push_back(pair<int,double>(x,EdgeProb_sn[i]));

	
	}
	adj_in.resize(n); 
	//Construct adjacency table (in)
	for(int j=0;j<InputEdges.size();j+=2){
		//edge x -> y
		x=InputEdges.at(j);
		y=InputEdges.at(j+1);
		int i=j/2;
		adj_in[y].push_back(pair<int,double>(x,EdgeProb[i])); 
	} 


	int num_nodes = adj_out.size();
    int total_degree = 0;
    int max_degree = 0;
 
    for (int i = 0; i < num_nodes; ++i) {
        int degree = adj_out[i].size() + adj_in[i].size();
        total_degree += degree;
        max_degree  = max(max_degree, degree);
    }
 
    double average_degree = static_cast<double>(total_degree) / num_nodes;
    cout<<"average_degree£º"<<average_degree<<endl;
	cout<<"max_degree:"<<max_degree<<endl; 
	
	
	//userID_Name;
	for(map<string, int>::iterator it_map=user2ID.begin();it_map!=user2ID.end();it_map++){
		ID2User[it_map->second]=it_map->first;
	}
	
	
	fprintf(stderr, "END reading graph (%s).\n", file); 
	
	

}

/* obtain social network and set parameters */
void readGraph2(const char* file,double gamma){
	//readfile, obtain vertexs edges
	int x,y;
	int count_edges=0;
	int count_vertices=0;
	string user_a,user_b;
	long double prob_edge;
	int flag=0;
	
	
	map<string, int>::iterator it;
	ifstream infile(file);
	
	while(infile>>user_a>>user_b>>flag) {
		if(flag==-1){
			it=foe_num.find(user_b);
			if(it!=foe_num.end()) foe_num[user_b]+=1;
			else foe_num[user_b]=1;
			continue;
		}
		
		//for a
		it=user2ID.find(user_a);
		if(it!=user2ID.end()){
			x=it->second;
		}
		else{
			username.push_back(user_a);
			user2ID[user_a]=count_vertices;
			x=count_vertices++;
		}
		
		//for b
		it=user2ID.find(user_b);
		if(it!=user2ID.end()){
			y=it->second;
			in_deg[user_b]+=1;

		}
		else{
			username.push_back(user_b);
			user2ID[user_b]=count_vertices;
			y=count_vertices++;
			in_deg[user_b]=1;

		}		
		
		
		
		count_edges++;
		InputEdges.push_back(x);
		InputEdges.push_back(y);	
	}
	
	n=count_vertices;
	m=count_edges;
	
	cout<<"number of users:"<<n<<" number of edges:"<<m<<"\n";
	
	//cost
	vector<string>::iterator itForCost;
	for(itForCost=username.begin();itForCost!=username.end();itForCost++){
		userCost[*itForCost]=1.0+gamma*in_deg[*itForCost];
	}

	 
	//viewpoint
	map<string, long double>::iterator it_for_pa;
	map<string, long double>::iterator it_for_pb;
	vector<string>::iterator it_vector;
	
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();  
    std::mt19937 gen(seed);  
  
 
    std::uniform_real_distribution<> dis(-1.0, 1.0);  
    

    std::uniform_real_distribution<> dis2(0, 1.0);  
    

	for(it_vector=username.begin();it_vector!=username.end();it_vector++)
	{
		double randomNumber = dis(gen);
		viewpoint[*it_vector]=round(randomNumber * 10) / 10;
	}
	
	//reputation
	vector<string>::iterator itForRepu;
	int cnt=0;
	for(itForRepu=username.begin();itForRepu!=username.end();itForRepu++){
		double randomNumber = dis2(gen);
		userReputation[*itForRepu]=1.0;
		it=foe_num.find(*itForRepu);
		if(it!=foe_num.end()){
			userReputation[*itForRepu]=(double)in_deg[*itForRepu]/(double)(in_deg[*itForRepu]+foe_num[*itForRepu]);
			cnt++;
		}
		

	}
	
	
	//EdgeProb
	ifstream infile2(file);
	while(infile2>>user_a>>user_b>>flag){
		if(flag==-1) continue;
		EdgeProb.push_back(1.0/in_deg[user_b]*(userReputation[user_a]+0.5) );
		EdgeProb_sn.push_back(1.0/in_deg[user_b]);
	}
	
	//NodeProb
	vector<string>::iterator itForNodeProb;
	for(itForNodeProb=username.begin();itForNodeProb!=username.end();itForNodeProb++){
		for(int l=0;l<21;l++){
			double puo=0.0;
			if(viewpoint[*itForNodeProb]>=-1.0+l*0.1) puo=1-(viewpoint[*itForNodeProb]+1.0-l*0.1)/2.0;
			else puo=1-(-1.0+l*0.1-viewpoint[*itForNodeProb])/2.0;
			
			NodeProb[*itForNodeProb].push_back(puo);
		}
		
	}	
	

	
	
	if(adj_out.size()<n) adj_out.resize(n);
	if(adj_out_reverse.size()<n) adj_out_reverse.resize(n);
	if(adj_sn_reverse.size()<n) adj_sn_reverse.resize(n);

	for (int j=0; j<InputEdges.size(); j+=2){
		//edge x -> y
		x = InputEdges.at(j);	 
		y = InputEdges.at(j+1);
		int i=j/2;
		adj_out[x].push_back(pair<int,double>(y,EdgeProb[i]));
		adj_out_reverse[y].push_back(pair<int,double>(x,EdgeProb[i]));
		adj_sn_reverse[y].push_back(pair<int,double>(x,EdgeProb_sn[i]));


	}
	adj_in.resize(n); 

	for(int j=0;j<InputEdges.size();j+=2){
		//edge x -> y
		x=InputEdges.at(j);
		y=InputEdges.at(j+1);
		int i=j/2;

		adj_in[y].push_back(pair<int,double>(x,EdgeProb[i])); 
	} 


	int num_nodes = adj_out.size();
    int total_degree = 0;
    int max_degree = 0;
 
    for (int i = 0; i < num_nodes; ++i) {
        int degree = adj_out[i].size() + adj_in[i].size();
        total_degree += degree;
        max_degree = max(max_degree, degree);
    }
 
    double average_degree = static_cast<double>(total_degree) / num_nodes;
    cout<<"average_degree£º"<<average_degree<<endl;
	cout<<"max_degree£º"<<max_degree<<endl; 

	//userID_Name;
	for(map<string, int>::iterator it_map=user2ID.begin();it_map!=user2ID.end();it_map++){
		ID2User[it_map->second]=it_map->first;
	}
	
	
	fprintf(stderr, "END reading graph (%s).\n", file); 
	
	

}



/*Diversity of Exposure of one user*/
double computeDiversityofExposure(set<double> L){
	double gu=0.0;
	int cnt=0;
	double prev=-1.0;
	for(auto it=L.begin();it!=L.end();it++){
		double current=*it;
		double diff=current-prev;
		gu+=pow(diff,2);
		prev=current;
		
	}
	
	gu+=pow(1.0-prev,2);
	
	double fu=1.0-gu/4.0;
	return fu;
	
}


/* compute total cost of a set of users*/
double computeUserCost(set<string> U){
	double sum=0.0;
	for(string u: U){
		sum+=userCost[u];
	}
	return sum;
}


/*RE set generation*/
pair<int, set<int> > generateREset(){
	

	set<int> REset;
	//generate v from V
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();  
    std::mt19937 gen(seed);   
    std::uniform_int_distribution<> dis(0, n - 1);    
    int v=dis(gen);	
	
	

  	
	//each o in O
	for(int o=0;o<o_len;o++){


		// queue
		
		queue<int> QT;
		QT.push(v);
		vector<int> visited(n,0);
		visited[v]=1;

		while(!QT.empty()){
			int q=QT.front();
			QT.pop();
			for(auto elem: adj_out_reverse[q]){
				if(a_rand()>elem.second) continue;
				int w=elem.first;
				if(a_rand()>NodeProb[ID2User[w]][o] || visited[w]==1) continue;
				QT.push(w);
				visited[w]=1;
				if(fabs(viewpoint[ID2User[w]]-opinions[o])<=1e-6) REset.insert(w);
			}	
		}		
		
	/*endfor*/
	}


	
	
	return std::make_pair(v, REset);
}

/*algorithms*/


double compute_F0(){
	double F0=0.0;
	vector<string>::iterator itForUser;
	for(itForUser=username.begin();itForUser!=username.end();itForUser++){
		set<double > L;
		L.insert(viewpoint[*itForUser]);
		F0+=computeDiversityofExposure(L);
	}
	F0=F0/(double)(n);
	return F0;
}

/*compute_theta_max_and_theta_0*/
void compute_theta_for_SEDM(double B,int& theta_max, int& theta_0, int &i_max,double &epsilon){
	
	//initial diversity
	double F0=compute_F0();

	//kMax
	int kMax = 0;
	vector<double> costs;
 	// Iterate over the map and push the values into the vector
    for (const auto& pair : userCost) {
        costs.push_back(pair.second);
    }
    
    // Sort the vector of costs in ascending order
    std::sort(costs.begin(), costs.end());
    
    // Initialize sum and count
    double sum = 0.0;
    
    // Iterate over the sorted costs and accumulate the sum
    for (double cost : costs) {
        sum += cost;
        kMax++;    
        // Check if the accumulated sum exceeds B
        if (sum > B) {
	        kMax-=1;
            break;
        }
    }


	//  1 - 1/e^beta
    double term1 = 1.0 - 1.0 / std::exp(beta);
 
	//  sqrt(ln(6) + ln(n^l))
    double term2 = std::sqrt(std::log(6) + l * std::log(n));
    
     //  sqrt((1 - 1/e^beta) * (ln(n^(k_max + 1 + l)) + ln(6)))
    double term3 = std::sqrt(term1 * (std::log(std::pow(n, kMax + 1 + l)) + std::log(6)));   
    
  
    double numerator = 2.0 * std::pow(term1 * term2 + term3, 2);
 
    double denominator = F0 * std::pow(epsilon, 2);
 

    double thetaMax = numerator / denominator;    

	double I_Max=log2(thetaMax/numerator);
	
	theta_max = static_cast<int>(ceil(thetaMax));
	theta_0 = static_cast<int>(ceil(numerator));
	i_max=static_cast<int>(std::ceil(I_Max));
}


//compute influence of u
double computeDelta(int& w, map<string, set<double> >& Lu, map<int ,set<int> >& nodeselect, int& number_R){
	
	string w_str=ID2User[w];
	double bw=userCost[w_str];
	double w_opinion=viewpoint[w_str];
	double total_u_level=0.0;
	
	for(int targetv: nodeselect[w]){
		string user_v= ID2User[targetv];
		set<double> L=Lu[user_v];
		double beforelevel=computeDiversityofExposure(L);
		L.insert(w_opinion);
		double afterlevel=computeDiversityofExposure(L);
		total_u_level+=(afterlevel-beforelevel)/bw;
	}
	
	
	return total_u_level/(double)number_R;
}


//compute influence of s
double computeDelta_s(int& w, map<string, set<double> >& Lu, map<int ,set<int> >& nodeselect, int& number_R){
	string w_str=ID2User[w];
	double bw=userCost[w_str];
	double w_opinion=viewpoint[w_str];
	double total_u_level=0.0;
	
	for(int targetv: nodeselect[w]){
		string user_v= ID2User[targetv];
		set<double> L=Lu[user_v];
		double beforelevel=computeDiversityofExposure(L);
		L.insert(w_opinion);
		double afterlevel=computeDiversityofExposure(L);
		total_u_level+=(afterlevel-beforelevel);
	}
	
	

	return total_u_level/(double)number_R;
}

double computeLowerBound(set<string>& S2, map<int, set<int> >& spreadAbility2, map<string, set<double> > Lu2, int& theta){
	double phi_R2=0.0;
	for(string w_str: S2){
		double w_opinion=viewpoint[w_str];
		int w=user2ID[w_str];
		for(int targetv: spreadAbility2[w]){
			string user_v= ID2User[targetv];
			double beforelevel=computeDiversityofExposure(Lu2[user_v]);
			Lu2[user_v].insert(w_opinion);
			double afterlevel=computeDiversityofExposure(Lu2[user_v]);
			phi_R2+=(afterlevel-beforelevel);
		}	
	}
	phi_R2=phi_R2/(double)theta;
	phi_R2+=compute_F0();
	
    double n_l = std::pow(n, -l);  //  n^(-l)
	double ln_3_over_n_l = std::log(3.0 / n_l);  //  ln(3 / n^(-l))	
	//  ¦Ò^l(S)
    double term1 = std::sqrt(theta * phi_R2 + (2.0 / 9.0) * ln_3_over_n_l);
	double term2 = std::sqrt(0.5 * ln_3_over_n_l);
	double result = ((term1 - term2) * (term1 - term2) - (1.0 / 18.0) * ln_3_over_n_l) / theta;
	
	return result;
	
}

double computeUpperBound(double& phi_R1, map<string, set<double> > Lu2, map<int ,set<int> >& nodeselect, int& theta, vector<string>& S_for_upper, double& B,double &epsilon){
	
	//compute phi_uR1;
	double phi_uR1=phi_R1/(the_ratio-epsilon);
	int l=S_for_upper.size();
	
	double diver_Si=compute_F0();
	for(int i=0;i<l;i++){
		//compute phi_R1_Si
		string w_str=S_for_upper[i];
		double w_opinion=viewpoint[w_str];
		int w=user2ID[w_str];
		for(int targetv: nodeselect[w]){
			string user_v= ID2User[targetv];
			double beforelevel=computeDiversityofExposure(Lu2[user_v]);
			Lu2[user_v].insert(w_opinion);
			double afterlevel=computeDiversityofExposure(Lu2[user_v]);
			diver_Si+=(afterlevel-beforelevel)/(double)theta;
		}
	
		//compute part2	
		double part2=0.0;
		
		map<int, double> weigain;

		for (const auto& mapPair : nodeselect) {
	        int w=mapPair.first;		
			string w_str=ID2User[w];
			double bw=userCost[w_str];
			double w_opinion=viewpoint[w_str];
			double w_for_u=0.0;
	        for (const auto& targetv : mapPair.second) {
	            string user_v= ID2User[targetv];
				set<double> L=Lu2[user_v];
				double beforelevel=computeDiversityofExposure(L);
				L.insert(w_opinion);
				double afterlevel=computeDiversityofExposure(L);
				w_for_u+=(afterlevel-beforelevel)/bw;
	        }
	        w_for_u=w_for_u/(double)theta;
	        weigain[w]=w_for_u;
    	}
 		double totalcost=0.0;
		int finalnode=-1;
		double finalPhi=0.0;
		while(totalcost<=B){
			double maxPhi=0.0;
			int maxnode=-1;
			for(const auto& mapPair: weigain){
				int w=mapPair.first;
				double w_for_u=mapPair.second;
				if(maxPhi<w_for_u){
					maxPhi=w_for_u;
					maxnode=w;
				}
			}
			string maxnode_str=ID2User[maxnode];
			if(totalcost+userCost[maxnode_str]<=B){
				totalcost+=userCost[maxnode_str];
				part2+=maxPhi;
			}
			else{
				finalnode=maxnode;
				finalPhi=maxPhi;
				break;
			}
		}
		part2+=finalPhi*(B-totalcost)/userCost[ID2User[finalnode]];
		
		if(phi_uR1>diver_Si+part2) phi_uR1=diver_Si+part2;
    	
		
	}

	
	double n_l = std::pow(n, -l);  //  n^(-l)
	double ln_3_over_n_l = std::log(3.0 / n_l);  //  ln(3 / n^(-l))
	
	//  ¦Ò^u(S^*)
	    
	double term1 = std::sqrt(theta * phi_uR1 + 0.5 * ln_3_over_n_l);
	  
	double term2 = std::sqrt(0.5 * ln_3_over_n_l);
	  
	double result = ((term1 + term2) * (term1 + term2)) / (double)theta;	
	
	return result;
}

//SEDM-
ResultType SEDM_OPIM(double B,double epsilon){

	auto start = std::chrono::high_resolution_clock::now();  
	
	
	set<string> finalS;
	double finalDiversity=0.0;
	
	int theta_max=0;
	int theta_0=0;
	int i_max=0;
	compute_theta_for_SEDM(B,theta_max,theta_0,i_max,epsilon);	//compute_theta_max_and_theta_0
	int theta=theta_0;
	vector<pair<int, std::set<int> > > R1,R2;

	//genereta RE sets;
	for(int t=0;t<theta;t++){
	
		pair<int, std::set<int>> reset1 = generateREset();
		pair<int, std::set<int>> reset2 = generateREset();
		R1.push_back(reset1);
		R2.push_back(reset2);
	}	


	
	for(int i=1;i<=i_max;i++){		

		map<int, set<int> >   spreadAbility, spreadAbility2;
		set<int> W, W2;
		for (const auto& pair_elem : R1) {
			int key = pair_elem.first;
	        const std::set<int>& values = pair_elem.second;
	      
	        for (int value : values) {
	        	W.insert(value);
	        	W2.insert(value);
	            spreadAbility[value].insert(key);
	        }
		}
		
		for (const auto& pair_elem : R2) {
			int key = pair_elem.first;
	        const std::set<int>& values = pair_elem.second;
	     
	        for (int value : values) {
	            spreadAbility2[value].insert(key);
	        }
		}		
		

		
		set<string> S1;
		vector<string> S_for_upper;
		
		//initial diversity
		double diver_S=compute_F0();
		
		map<string, set<double> > Lu,Lu2;
		
		for(string u: username){
			Lu[u].insert(viewpoint[u]);
			Lu[u].insert(1);
			Lu[u].insert(-1);
			Lu2[u].insert(viewpoint[u]);
			Lu2[u].insert(1);
			Lu2[u].insert(-1);
		}
			
		while(!W.empty()){		
			//compute maximal u
			int u=-1;
			double maximalDelta=-1.0;
			for(int w : W){
				if(B-userCost[ID2User[w]]-computeUserCost(S1)<0) continue;
				double delt=computeDelta(w,Lu,spreadAbility,theta);
				
				if(delt-maximalDelta>=0){
					u=w;
					maximalDelta=delt;
				}	
			
			}
			if(u==-1)break;

		
			//delete u from W    
			string u_str=ID2User[u];
    		auto it = std::find(W.begin(), W.end(), u);    
    		if (it != W.end()) W.erase(it);
			
			//add u to S'
			if(B-userCost[u_str]-computeUserCost(S1)>=0){
				S1.insert(u_str);
				S_for_upper.push_back(u_str);
				diver_S+=maximalDelta*userCost[u_str];
				//update Lu
				for(int v: spreadAbility[u]){
					string v_str=ID2User[v];
					Lu[v_str].insert(viewpoint[u_str]);
				}
				
					
			}			
		}
				
		
		
		double diver_s=0.0;
		int s=-1;
		//maximal s
		for(int w:W2){
			double delt=computeDelta_s(w,Lu2,spreadAbility,theta);
			if(diver_s<delt){
				diver_s=delt;
				s=w;
			}
		}
		diver_s+=compute_F0(); 
		
		set<string> S2;
		if(diver_s>diver_S){
			S2.insert(ID2User[s]);
			diver_S=diver_s;
		}
		else S2=S1;
		
		
		double diverLowerBound=computeLowerBound(S2,spreadAbility2,Lu2,theta);
		
		double diverupperBound=computeUpperBound(diver_S,Lu2,spreadAbility,theta,S_for_upper,B,epsilon);
		

		if(diverLowerBound/diverupperBound>=the_ratio-epsilon){
			finalDiversity=diver_S;
			finalS=S2;
			break;
		}
		
		for(int t=0;t<theta;t++){
			pair<int, std::set<int>> reset1 = generateREset();
			pair<int, std::set<int>> reset2 = generateREset();
			R1.push_back(reset1);
			R2.push_back(reset2);	
		}		
		theta=theta*2;

	}
	
	auto end = std::chrono::high_resolution_clock::now();  
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);  
	
    cout<<"SEDM_OPIM_Diversity: "<<finalDiversity<<endl;
    cout << "SEDM_OPIM_Running time: " << duration.count()*1e-3 << "s" << std::endl;
    cout<<"Size of Seed set: "<< finalS.size()<<endl;
    
    ResultType res_return;
    res_return.name="SEDM_OPIM";
    res_return.finaldiver=finalDiversity;
    res_return.runningtime=duration.count()*1e-3;
    res_return.seedsize=finalS.size();
    return res_return;
}


//SEDM 
ResultType SEDM(double B,double epsilon){

	auto start = std::chrono::high_resolution_clock::now();  
	
	
	set<string> finalS;
	double finalDiversity=0.0;
	
	int theta_max=0;
	int theta_0=0;
	int i_max=0;
	compute_theta_for_SEDM(B,theta_max,theta_0,i_max,epsilon);	//compute_theta_max_and_theta_0
	int theta=theta_0;
	vector<pair<int, std::set<int> > > R1,R2;

	//genereta RE sets;
	for(int t=0;t<theta;t++){
	
		pair<int, std::set<int>> reset1 = generateREset();
		pair<int, std::set<int>> reset2 = generateREset();
		R1.push_back(reset1);
		R2.push_back(reset2);
	}	


	
	for(int i=1;i<=i_max;i++){		
		map<int, set<int> >   spreadAbility, spreadAbility2;
		set<int> W, W2;
		for (const auto& pair_elem : R1) {
			int key = pair_elem.first;
	        const std::set<int>& values = pair_elem.second;
	  
	        for (int value : values) {
	        	W.insert(value);
	        	W2.insert(value);
	            spreadAbility[value].insert(key);
	        }
		}
		
		for (const auto& pair_elem : R2) {
			int key = pair_elem.first;
	        const std::set<int>& values = pair_elem.second;
	      
	        for (int value : values) {
	            spreadAbility2[value].insert(key);
	        }
		}		
		

		set<string> S1;
		vector<string> S_for_upper;

		//initial diversity
		double diver_S=compute_F0();
		
		map<string, set<double> > Lu,Lu2;
		
		for(string u: username){
			Lu[u].insert(viewpoint[u]);
			Lu[u].insert(1);
			Lu[u].insert(-1);
			Lu2[u].insert(viewpoint[u]);
			Lu2[u].insert(1);
			Lu2[u].insert(-1);
		}
			
		while(!W.empty()){		
			//compute maximal u
			int u=-1;
			double maximalDelta=-1.0;
			for(int w : W){
				if(B-userCost[ID2User[w]]-computeUserCost(S1)<0) continue;
				double delt=computeDelta(w,Lu,spreadAbility,theta);
				
				if(delt-maximalDelta>=0){
					u=w;
					maximalDelta=delt;
				}	
			
			}
			if(u==-1)break;

		
			//delete u from W    
			string u_str=ID2User[u];
    		auto it = std::find(W.begin(), W.end(), u);    
    		if (it != W.end()) W.erase(it);
			
			//add u to S'
			if(B-userCost[u_str]-computeUserCost(S1)>=0){
				S1.insert(u_str);
				S_for_upper.push_back(u_str);
				diver_S+=maximalDelta*userCost[u_str];
				//update Lu
				for(int v: spreadAbility[u]){
					string v_str=ID2User[v];
					Lu[v_str].insert(viewpoint[u_str]);
				}
				
					
			}			
		}
				
		
		double diver_s=0.0;
		int s=-1;
		//maximal s
		for(int w:W2){
			double delt=computeDelta_s(w,Lu2,spreadAbility,theta);
			if(diver_s<delt){
				diver_s=delt;
				s=w;
			}
		}
		diver_s+=compute_F0(); 
		
		set<string> S2;
		if(diver_s>diver_S){
			S2.insert(ID2User[s]);
			diver_S=diver_s;
		}
		else S2=S1;
		
		double diverLowerBound=computeLowerBound(S2,spreadAbility2,Lu2,theta);
		
		double diverupperBound=computeUpperBound(diver_S,Lu2,spreadAbility,theta,S_for_upper,B,epsilon);
		

		if(diverLowerBound/diverupperBound>=the_ratio-epsilon){
			finalDiversity=diver_S;
			finalS=S2;
			break;
		}
		
		//  R1 -> set 
		std::set<std::pair<int, std::set<int>>> R1_set(R1.begin(), R1.end());
		
		// R1 \cup R2
		for (const auto& p : R2) {
		    R1_set.insert(p);
		}
			
		// R1 -> vector
		R1.assign(R1_set.begin(), R1_set.end());
		
		 
		
		for(int t=0;t<theta;t++){
			if(R1.size()<theta*2){
				pair<int, std::set<int>> reset1 = generateREset();
				R1.push_back(reset1);
			}
			
			pair<int, std::set<int>> reset2 = generateREset();
			R2.push_back(reset2);	
		}		
		theta=theta*2;

	}
	
	auto end = std::chrono::high_resolution_clock::now();  
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); 
	
    cout<<"SEDM_Diversity: "<<finalDiversity<<endl;
    cout << "SEDM_Running time: " << duration.count()*1e-3 << "s" << std::endl;
    cout<<"Size of Seed set: "<< finalS.size()<<endl;
    
    ResultType res_return;
    res_return.name="SEDM";
    res_return.finaldiver=finalDiversity;
    res_return.runningtime=duration.count()*1e-3;
    res_return.seedsize=finalS.size();
    return res_return;
}

//EDM+
ResultType IMM_EDM(double B,double epsilon){
	
	auto start = std::chrono::high_resolution_clock::now(); 

	set<string> finalS;
	double finalDiversity=0.0;
	
	int theta_max=0;
	int theta_0=0;
	int i_max=0;
	compute_theta_for_SEDM(B,theta_max,theta_0,i_max,epsilon);	//compute_theta_max
	
	int theta=theta_max;  //IMM 
	vector<pair<int, std::set<int> > > R1,R2;

	//genereta RE sets;
	for(int t=0;t<theta;t++){
		pair<int, std::set<int>> reset1 = generateREset();
		R1.push_back(reset1);	
	}
	

	map<int, set<int> >   spreadAbility, spreadAbility2;
	set<int> W, W2;
	for (const auto& pair_elem : R1) {
		int key = pair_elem.first;
        const std::set<int>& values = pair_elem.second;
     
        for (int value : values) {
        	W.insert(value);
        	W2.insert(value);
            spreadAbility[value].insert(key);
        }
	}
	
	for (const auto& pair_elem : R2) {
		int key = pair_elem.first;
        const std::set<int>& values = pair_elem.second;
     
        for (int value : values) {
            spreadAbility2[value].insert(key);
        }
	}		
	
	set<string> S1;
	vector<string> S_for_upper;
	
	//initial diversity
	double diver_S=compute_F0();
	
	map<string, set<double> > Lu,Lu2;
	
	for(string u: username){
		Lu[u].insert(viewpoint[u]);
		Lu[u].insert(1);
		Lu[u].insert(-1);
		Lu2[u].insert(viewpoint[u]);
		Lu2[u].insert(1);
		Lu2[u].insert(-1);
	}
		
	while(!W.empty()){		
		//compute maximal u
		int u=-1;
		double maximalDelta=-1.0;
		for(int w : W){
			if(B-userCost[ID2User[w]]-computeUserCost(S1)<0) continue;
			double delt=computeDelta(w,Lu,spreadAbility,theta);
			
			if(delt-maximalDelta>=0){
				u=w;
				maximalDelta=delt;
			}	
		
		}
		if(u==-1)break;

	
		//delete u from W    
		string u_str=ID2User[u];
		auto it = std::find(W.begin(), W.end(), u);    
		if (it != W.end()) W.erase(it);
		
		//add u to S'
		if(B-userCost[u_str]-computeUserCost(S1)>=0){
			S1.insert(u_str);
			S_for_upper.push_back(u_str);
			diver_S+=maximalDelta*userCost[u_str];
			//update Lu
			for(int v: spreadAbility[u]){
				string v_str=ID2User[v];
				Lu[v_str].insert(viewpoint[u_str]);
			}	
				
		}			
	}
			

	
	
	double diver_s=0.0;
	int s=-1;
	//maximal s
	for(int w:W2){
		double delt=computeDelta_s(w,Lu2,spreadAbility,theta);
		if(diver_s<delt){
			diver_s=delt;
			s=w;
		}
	}
	diver_s+=compute_F0(); 
	
	set<string> S2;
	if(diver_s>diver_S){
		S2.insert(ID2User[s]);
		diver_S=diver_s;
	}
	else S2=S1;
				
	finalDiversity=diver_S;
	finalS=S2;
	
	auto end = std::chrono::high_resolution_clock::now();  
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);  	
	
	
	cout<<"IMM_Diversity: "<<finalDiversity<<endl;
    cout << "IMM_Running time: " << duration.count()*1e-3 << "s" << std::endl;
    cout<<"Size of Seed set: "<< finalS.size()<<endl;
    ResultType res_return;
    res_return.name="IMM_IEDM";
    res_return.finaldiver=finalDiversity;
    res_return.runningtime=duration.count()*1e-3;
    res_return.seedsize=finalS.size();
    return res_return;


}

//WOC
ResultType WOC(double B,double epsilon){

	auto start = std::chrono::high_resolution_clock::now();  
	
	
	set<string> finalS;
	double finalDiversity=0.0;
	
	int theta_max=0;
	int theta_0=0;
	int i_max=0;
	compute_theta_for_SEDM(B,theta_max,theta_0,i_max,epsilon);	//compute_theta_max_and_theta_0
	int theta=theta_0;
	vector<pair<int, std::set<int> > > R1,R2;

	//genereta RE sets;
	for(int t=0;t<theta;t++){
	
		pair<int, std::set<int>> reset1 = generateREset();
		pair<int, std::set<int>> reset2 = generateREset();
		R1.push_back(reset1);
		R2.push_back(reset2);
		
	}	


	
	for(int i=1;i<=i_max;i++){		
	
		map<int, set<int> >   spreadAbility, spreadAbility2;
		set<int> W, W2;
		for (const auto& pair_elem : R1) {
			int key = pair_elem.first;
	        const std::set<int>& values = pair_elem.second;
	        
	        for (int value : values) {
	        	W.insert(value);
	        	W2.insert(value);
	            spreadAbility[value].insert(key);
	        }
		}
		
		for (const auto& pair_elem : R2) {
			int key = pair_elem.first;
	        const std::set<int>& values = pair_elem.second;
	        
	        for (int value : values) {
	            spreadAbility2[value].insert(key);
	        }
		}		
		
		set<string> S1;
		vector<string> S_for_upper;
		
		//initial diversity
		double diver_S=compute_F0();
		
		map<string, set<double> > Lu,Lu2;
		
		for(string u: username){
			Lu[u].insert(viewpoint[u]);
			Lu[u].insert(1);
			Lu[u].insert(-1);
			Lu2[u].insert(viewpoint[u]);
			Lu2[u].insert(1);
			Lu2[u].insert(-1);
		}
			
		while(!W.empty()){		
			//compute maximal u
			int u=-1;
			double maximalDelta=-1.0;
			for(int w : W){
				if(B-userCost[ID2User[w]]-computeUserCost(S1)<0) continue;
				double delt=computeDelta(w,Lu,spreadAbility,theta)*userCost[ID2User[w]];
				
				if(delt-maximalDelta>=0){
					u=w;
					maximalDelta=delt;
				}	
			
			}
			if(u==-1)break;

		
			//delete u from W    
			string u_str=ID2User[u];
    		auto it = std::find(W.begin(), W.end(), u);    
    		if (it != W.end()) W.erase(it);
			
			//add u to S'
			if(B-userCost[u_str]-computeUserCost(S1)>=0){
				S1.insert(u_str);
				S_for_upper.push_back(u_str);
				diver_S+=maximalDelta;
				//update Lu
				for(int v: spreadAbility[u]){
					string v_str=ID2User[v];
					Lu[v_str].insert(viewpoint[u_str]);
				}
				
					
			}			
		}

		
		double diver_s=0.0;
		int s=-1;
		//maximal s
		for(int w:W2){
			double delt=computeDelta_s(w,Lu2,spreadAbility,theta);
			if(diver_s<delt){
				diver_s=delt;
				s=w;
			}
		}
		diver_s+=compute_F0(); 
		
		set<string> S2;
		if(diver_s>diver_S){
			S2.insert(ID2User[s]);
			diver_S=diver_s;
		}
		else S2=S1;
		
		
		double diverLowerBound=computeLowerBound(S2,spreadAbility2,Lu2,theta);
		
		double diverupperBound=computeUpperBound(diver_S,Lu2,spreadAbility,theta,S_for_upper,B,epsilon);
		
		if(diverLowerBound/diverupperBound>=the_ratio-epsilon){
			finalDiversity=diver_S;
			finalS=S2;
			break;
		}
		
		for(int t=0;t<theta;t++){
			pair<int, std::set<int>> reset1 = generateREset();
			pair<int, std::set<int>> reset2 = generateREset();
			R1.push_back(reset1);
			R2.push_back(reset2);	
		}		
		theta=theta*2;

	}

	auto end = std::chrono::high_resolution_clock::now();  
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);   	
	
    cout<<"WOC_Diversity: "<<finalDiversity<<endl;
    cout << "WOC_Running time: " << duration.count()*1e-3 << "s" << std::endl;	
    cout<<"Size of Seed set: "<< finalS.size()<<endl;

    ResultType res_return;
    res_return.name="WOC";
    res_return.finaldiver=finalDiversity;
    res_return.runningtime=duration.count()*1e-3;
    res_return.seedsize=finalS.size();
    return res_return;
	
} 


//generateREs for WOR 
pair<int, set<int> > generateREs_SN(){
	

	set<int> REset;
	//generate v from V
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();  
    std::mt19937 gen(seed);   
    std::uniform_int_distribution<> dis(0, n - 1);    
    int v=dis(gen);	
	
	

  	
	//each o in O
	for(int o=0;o<o_len;o++){


		// queue
		
		queue<int> QT;
		QT.push(v);
		vector<int> visited(n,0);
		visited[v]=1;

		while(!QT.empty()){
			int q=QT.front();
			QT.pop();
			for(auto elem: adj_sn_reverse[q]){
				if(a_rand()>elem.second) continue;
				int w=elem.first;
				if(a_rand()>NodeProb[ID2User[w]][o] || visited[w]==1) continue;
				QT.push(w);
				visited[w]=1;
				if(fabs(viewpoint[ID2User[w]]-opinions[o])<=1e-6) REset.insert(w);
			}	
		}		
		
	/*endfor*/
	}


	
	
	return std::make_pair(v, REset);
}

vector<pair<int,set<int> > > generateREs_SN_SNR(){

	set<int> REset;
	set<int> REset_sn;
	//generate v from V
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();  
    std::mt19937 gen(seed);   
    std::uniform_int_distribution<> dis(0, n - 1);    
    int v=dis(gen);	
	
  	
	//each o in O
	for(int o=0;o<o_len;o++){


		// queue
		
		queue<int> QT;
		QT.push(v);
		vector<int> visited(n,0);
		visited[v]=1;

		while(!QT.empty()){
			int q=QT.front();
			QT.pop();
			for(auto elem: adj_out_reverse[q]){
				if(a_rand()>elem.second) continue;
				int w=elem.first;
				if(a_rand()>NodeProb[ID2User[w]][o] || visited[w]==1) continue;
				QT.push(w);
				visited[w]=1;
				if(fabs(viewpoint[ID2User[w]]-opinions[o])<=1e-6) REset.insert(w);
			}	
		}		
		
	/*endfor*/
	}

	//each o in O
	for(int o=0;o<o_len;o++){


		// queue
		
		queue<int> QT;
		QT.push(v);
		vector<int> visited(n,0);
		visited[v]=1;

		while(!QT.empty()){
			int q=QT.front();
			QT.pop();
			for(auto elem: adj_sn_reverse[q]){
				if(a_rand()>elem.second) continue;
				int w=elem.first;
				if(a_rand()>NodeProb[ID2User[w]][o] || visited[w]==1) continue;
				QT.push(w);
				visited[w]=1;
				if(fabs(viewpoint[ID2User[w]]-opinions[o])<=1e-6) REset_sn.insert(w);
			}	
		}		
		
	/*endfor*/
	}
	
	pair<int,set<int>> p1=std::make_pair(v, REset);
	pair<int,set<int>> p2=std::make_pair(v, REset_sn);
	vector<pair<int,set<int> > > vec;
	vec.push_back(p1);
	vec.push_back(p2);
	
	
	return vec;	
}

//WOR
ResultType WOR(double B,double epsilon){
	auto start = std::chrono::high_resolution_clock::now();  
	
	
	set<string> finalS;
	double finalDiversity=0.0;
	
	int theta_max=0;
	int theta_0=0;
	int i_max=0;
	compute_theta_for_SEDM(B,theta_max,theta_0,i_max,epsilon);	//compute_theta_max_and_theta_0
	int theta=theta_0;
	vector<pair<int, std::set<int> > > R1,R2,R1_snr;

	//genereta RE sets;
	for(int t=0;t<theta;t++){

		vector<pair<int,set<int> > > reset1 = generateREs_SN_SNR();

		pair<int, std::set<int>> reset2 = generateREs_SN();

		R1_snr.push_back(reset1[0]);
		R1.push_back(reset1[1]);
		R2.push_back(reset2);
	
	}	

	

		
	for(int i=1;i<=i_max;i++){		

		map<int, set<int> >   spreadAbility, spreadAbility2;
		set<int> W, W2;
		for (const auto& pair_elem : R1) {
			int key = pair_elem.first;
	        const std::set<int>& values = pair_elem.second;

	        for (int value : values) {
	        	W.insert(value);
	        	W2.insert(value);
	            spreadAbility[value].insert(key);
	        }
		}
		
		for (const auto& pair_elem : R2) {
			int key = pair_elem.first;
	        const std::set<int>& values = pair_elem.second;

	        for (int value : values) {
	            spreadAbility2[value].insert(key);
	        }
		}		
		
		set<string> S1;
		vector<string> S_for_upper;
		
		//initial diversity
		double diver_S=compute_F0();
		
		map<string, set<double> > Lu,Lu2;
		
		for(string u: username){
			Lu[u].insert(viewpoint[u]);
			Lu[u].insert(1);
			Lu[u].insert(-1);
			Lu2[u].insert(viewpoint[u]);
			Lu2[u].insert(1);
			Lu2[u].insert(-1);
		}
			
		while(!W.empty()){		
			//compute maximal u
			int u=-1;
			double maximalDelta=-1.0;
			for(int w : W){
				if(B-userCost[ID2User[w]]-computeUserCost(S1)<0) continue;
				double delt=computeDelta(w,Lu,spreadAbility,theta);
				
				if(delt-maximalDelta>=0){
					u=w;
					maximalDelta=delt;
				}	
			
			}
			if(u==-1)break;

		
			//delete u from W    
			string u_str=ID2User[u];
    		auto it = std::find(W.begin(), W.end(), u);    
    		if (it != W.end()) W.erase(it);
			
			//add u to S'
			if(B-userCost[u_str]-computeUserCost(S1)>=0){
				S1.insert(u_str);
				S_for_upper.push_back(u_str);
				diver_S+=maximalDelta*userCost[u_str];
				//update Lu
				for(int v: spreadAbility[u]){
					string v_str=ID2User[v];
					Lu[v_str].insert(viewpoint[u_str]);
				}
				
					
			}			
		}
				
		
		double diver_s=0.0;
		int s=-1;
		//maximal s
		for(int w:W2){
			double delt=computeDelta_s(w,Lu2,spreadAbility,theta);
			if(diver_s<delt){
				diver_s=delt;
				s=w;
			}
		}
		diver_s+=compute_F0(); 
		
		set<string> S2;
		if(diver_s>diver_S){
			S2.insert(ID2User[s]);
			diver_S=diver_s;
		}
		else S2=S1;
		
		
		double diverLowerBound=computeLowerBound(S2,spreadAbility2,Lu2,theta);
		
		double diverupperBound=computeUpperBound(diver_S,Lu2,spreadAbility,theta,S_for_upper,B,epsilon);
		

		if(diverLowerBound/diverupperBound>=the_ratio-epsilon){
			finalDiversity=diver_S;
			finalS=S2;
			break;
		}
		
		for(int t=0;t<theta;t++){
			vector<pair<int,set<int> > > reset1 = generateREs_SN_SNR();
			
			
			pair<int, std::set<int>> reset2 = generateREs_SN();
			R1_snr.push_back(reset1[0]);
			R1.push_back(reset1[1]);
			R2.push_back(reset2);	
		}		
		theta=theta*2;

	}

	auto end = std::chrono::high_resolution_clock::now();  
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);  	
	
	finalDiversity=0.0;
	
	map<int, set<int> >   spreadAbility;

	for (const auto& pair_elem : R1_snr) {
		int key = pair_elem.first;
        const std::set<int>& values = pair_elem.second;
        for (int value : values) {

            spreadAbility[value].insert(key);
        }
	}
	
		
	//initial diversity
	double diver_S=compute_F0();
	
	map<string, set<double> > Lu;
	
	for(string u: username){
		Lu[u].insert(viewpoint[u]);
		Lu[u].insert(1);
		Lu[u].insert(-1);
	}

	
	for(string s_str: finalS){
		int s=user2ID[s_str];
		double delt=computeDelta(s,Lu,spreadAbility,theta);
		diver_S+=delt*userCost[ID2User[s]];
		for(int v: spreadAbility[s]){
			string v_str=ID2User[v];
			Lu[v_str].insert(viewpoint[ID2User[s]]);
		}
	}
	
	finalDiversity=diver_S;

		
	
    cout<<"WOR_Diversity: "<<finalDiversity<<endl;
    cout << "WOR_Running time: " << duration.count()*1e-3 << "s" << std::endl;   
    cout<<"Size of Seed set: "<< finalS.size()<<endl;

    ResultType res_return;
    res_return.name="WOR";
    res_return.finaldiver=finalDiversity;
    res_return.runningtime=duration.count()*1e-3;
    res_return.seedsize=finalS.size();
    return res_return;

}

//OD
ResultType OutDegree(double B,double epsilon){

	auto start = std::chrono::high_resolution_clock::now();  

    vector<pair<int, int>> outDegrees; // To store node and its out-degree
 
    // Calculate out-degree for each node
    for (size_t i = 0; i < adj_out.size(); ++i) {
        int outDegree = adj_out[i].size();
        outDegrees.push_back(make_pair(i, outDegree));
    }
 
    // Sort nodes by out-degree in descending order
    sort(outDegrees.begin(), outDegrees.end(), [](const pair<int, int>& a, const pair<int, int>& b) {
        return a.second > b.second;
    });
 
    // Extract the top k nodes with the highest out-degree
    vector<int> topNodes;
    double totalcost=0.0; 
    for (int i = 0; i < outDegrees.size(); ++i) {
        if(totalcost+userCost[ID2User[outDegrees[i].first]]>B) continue;
		topNodes.push_back(outDegrees[i].first);
		totalcost+=userCost[ID2User[outDegrees[i].first]];
    }	

	
 	auto end = std::chrono::high_resolution_clock::now();  
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); 
	
	set<string> finalS;
	double finalDiversity=0.0;
	
	int theta_max=0;
	int theta_0=0;
	int i_max=0;
	compute_theta_for_SEDM(B,theta_max,theta_0,i_max,epsilon);	//compute_theta_max
	
	int theta=theta_max;  //IMM 
	vector<pair<int, std::set<int> > > R1,R2;

	//genereta RE sets;
	for(int t=0;t<theta;t++){
	
		pair<int, std::set<int>> reset1 = generateREset();
		R1.push_back(reset1);
		
	}
	
	map<int, set<int> >   spreadAbility, spreadAbility2;
	set<int> W, W2;
	for (const auto& pair_elem : R1) {
		int key = pair_elem.first;
        const std::set<int>& values = pair_elem.second;

        for (int value : values) {
        	W.insert(value);
        	W2.insert(value);
            spreadAbility[value].insert(key);
        }
	}
	
	for (const auto& pair_elem : R2) {
		int key = pair_elem.first;
        const std::set<int>& values = pair_elem.second;

        for (int value : values) {
            spreadAbility2[value].insert(key);
        }
	}		
	

	set<string> S1;
	vector<string> S_for_upper;
	
	//initial diversity
	double diver_S=compute_F0();
	
	map<string, set<double> > Lu,Lu2;
	
	for(string u: username){
		Lu[u].insert(viewpoint[u]);
		Lu[u].insert(1);
		Lu[u].insert(-1);
		Lu2[u].insert(viewpoint[u]);
		Lu2[u].insert(1);
		Lu2[u].insert(-1);
	}

	
	for(int s: topNodes){
		double delt=computeDelta(s,Lu,spreadAbility,theta);
		diver_S+=delt*userCost[ID2User[s]];
		for(int v: spreadAbility[s]){
			string v_str=ID2User[v];
			Lu[v_str].insert(viewpoint[ID2User[s]]);
		}
	}
	
	finalDiversity=diver_S;

	
	
    cout<<"OutDegree_Diversity: "<<finalDiversity<<endl;
    cout << "OutDegree_Running time: " << duration.count()*1e-3 << "s" << std::endl;   
    cout<<"Size of Seed set: "<< topNodes.size()<<endl;
    
    ResultType res_return;
    res_return.name="OD";
    res_return.finaldiver=finalDiversity;
    res_return.runningtime=duration.count()*1e-3;
    res_return.seedsize=topNodes.size();
    return res_return;    
    
}

//PR
ResultType PageRank(double B,double epsilon){
	auto start = std::chrono::high_resolution_clock::now();  	
	
	const double DAMPING_FACTOR = 0.85;
	const double EPSILON = 1e-6;
	vector<double> pageRank(n, 1.0 / n);
	
	vector<double> outDegreeSum(n, 0);
        
    // Calculate the sum of out-degrees (weights) for each node
    for (int u = 0; u < n; ++u) {
        for (const auto& edge : adj_out[u]) {
            outDegreeSum[u] += edge.second;
        }
    }

    bool converged = false;
    while (!converged) {
        vector<double> newPageRank(n, 0);
        
        for (int u = 0; u < n; ++u) {
            if (outDegreeSum[u] == 0) continue; // Handle dangling nodes
            
            for (const auto& edge : adj_out[u]) {
                int v = edge.first;
                double weight = edge.second;
                newPageRank[v] += pageRank[u] * weight / outDegreeSum[u];
            }
        }

        double sum = 0;
        for (int u = 0; u < n; ++u) {
            newPageRank[u] = (1 - DAMPING_FACTOR) / n + DAMPING_FACTOR * newPageRank[u];
            sum += fabs(newPageRank[u] - pageRank[u]);
            pageRank[u] = newPageRank[u];
        }

        if (sum < EPSILON) {
            converged = true;
        }
    }
    
    vector<pair<double, int>> valueIndexPairs;
	for (int i = 0; i < pageRank.size(); ++i) {
	    valueIndexPairs.push_back({pageRank[i], i});
	}
	// sort
   
	sort(valueIndexPairs.begin(), valueIndexPairs.end(), [](const pair<double, int>& a, const pair<double, int>& b) {        
		return a.first > b.first;
	});
   
	

    vector<int> topNodes;
   	double totalcost=0.0;
	for (int i = 0;  i < valueIndexPairs.size(); ++i) {
		if(totalcost+userCost[ID2User[i]]>B) continue;
        topNodes.push_back(valueIndexPairs[i].second);
        totalcost+=userCost[ID2User[i]];
    }
    
 	auto end = std::chrono::high_resolution_clock::now();  
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);    
    
	set<string> finalS;
	double finalDiversity=0.0;
	
	int theta_max=0;
	int theta_0=0;
	int i_max=0;
	compute_theta_for_SEDM(B,theta_max,theta_0,i_max,epsilon);	//compute_theta_max
	
	int theta=theta_max;  //IMM 
	vector<pair<int, std::set<int> > > R1,R2;

	//genereta RE sets;
	for(int t=0;t<theta;t++){
	
		pair<int, std::set<int>> reset1 = generateREset();
		R1.push_back(reset1);
		
	}
	
	map<int, set<int> >   spreadAbility, spreadAbility2;
	set<int> W, W2;
	for (const auto& pair_elem : R1) {
		int key = pair_elem.first;
        const std::set<int>& values = pair_elem.second;

        for (int value : values) {
        	W.insert(value);
        	W2.insert(value);
            spreadAbility[value].insert(key);
        }
	}
	
	for (const auto& pair_elem : R2) {
		int key = pair_elem.first;
        const std::set<int>& values = pair_elem.second;
 
        for (int value : values) {
            spreadAbility2[value].insert(key);
        }
	}		
	
	
	set<string> S1;
	vector<string> S_for_upper;

	//initial diversity
	double diver_S=compute_F0();
	
	map<string, set<double> > Lu,Lu2;
	
	for(string u: username){
		Lu[u].insert(viewpoint[u]);
		Lu[u].insert(1);
		Lu[u].insert(-1);
		Lu2[u].insert(viewpoint[u]);
		Lu2[u].insert(1);
		Lu2[u].insert(-1);
	}

	
	for(int s: topNodes){
		double delt=computeDelta(s,Lu,spreadAbility,theta);
		diver_S+=delt*userCost[ID2User[s]];
		for(int v: spreadAbility[s]){
			string v_str=ID2User[v];
			Lu[v_str].insert(viewpoint[ID2User[s]]);
		}
	}
	
	finalDiversity=diver_S;
	
	
    cout<<"PageRank_Diversity: "<<finalDiversity<<endl;
    cout << "PageRank_Running time: " << duration.count()*1e-3 << "s" << std::endl;       
    cout<<"Size of Seed set: "<< topNodes.size()<<endl;
    
    ResultType res_return;
    res_return.name="PageRank";
    res_return.finaldiver=finalDiversity;
    res_return.runningtime=duration.count()*1e-3;
    res_return.seedsize=topNodes.size();
    return res_return;     
    
}


void menu(string folder,double epsilon){

	vector<vector<ResultType> >  vec_res(7);

	double B=10.0;
	for(int i=1;i<=5;i++){
		B=10.0*i;
		ResultType r_sedm=SEDM(B,epsilon);
		ResultType r_opim=SEDM_OPIM(B,epsilon);
		ResultType r_imm_edm=IMM_EDM(B,epsilon);
		ResultType r_woc=WOC(B,epsilon);
		ResultType r_wor=WOR(B,epsilon);
		ResultType r_od=OutDegree(B,epsilon);
		ResultType r_pr=PageRank(B,epsilon);
		
		vec_res[0].push_back(r_sedm);
		vec_res[1].push_back(r_woc);
		vec_res[2].push_back(r_wor);
		vec_res[3].push_back(r_opim);
		vec_res[4].push_back(r_imm_edm);
		vec_res[5].push_back(r_od);
		vec_res[6].push_back(r_pr);
		
	}	
	
	//write effectiveness_B
	string myfolder=folder;
	string filename="effectiveness_new.txt";
	string filePath=myfolder+"/"+filename;
	
	std::ofstream outFile(filePath);
	if (!outFile) {
    	std::cerr << "Error opening file: " << filePath << std::endl;
	}
    
    std::vector<std::tuple<std::string, double, double, double, double, double>> data_effect;
    data_effect.push_back(make_tuple("B",10.0,20.0,30.0,40.0,50.0));
    for(int i=0;i<7;i++){
    	data_effect.push_back( make_tuple(vec_res[i][0].name,vec_res[i][0].finaldiver,vec_res[i][1].finaldiver,vec_res[i][2].finaldiver,vec_res[i][3].finaldiver,vec_res[i][4].finaldiver) );
	}
		
    outFile << std::fixed << std::setprecision(5);

    // write -> file
    for (const auto& row : data_effect) {
        outFile << std::get<0>(row) << " "
                << std::get<1>(row) << " "
                << std::get<2>(row) << " "
                << std::get<3>(row) << " "
                << std::get<4>(row) << " "
                << std::get<5>(row) << "\n";
    }

    // close
    outFile.close();


	//write effciency_B
	filename="effciency_new.txt";
	filePath=myfolder+"/"+filename;
	
	std::ofstream outFile2(filePath);
	if (!outFile2) {
    	std::cerr << "Error opening file: " << filePath << std::endl;
	}
    
    std::vector<std::tuple<std::string, double, double, double, double, double>> data_effciency;
    data_effciency.push_back(make_tuple("B",10.0,20.0,30.0,40.0,50.0));
    for(int i=0;i<7;i++){
    	data_effciency.push_back( make_tuple(vec_res[i][0].name,vec_res[i][0].runningtime,vec_res[i][1].runningtime,vec_res[i][2].runningtime,vec_res[i][3].runningtime,vec_res[i][4].runningtime) );
	}
				

    outFile2 << std::fixed << std::setprecision(3);

    for (const auto& row : data_effciency) {
        outFile2 << std::get<0>(row) << " "
                << std::get<1>(row) << " "
                << std::get<2>(row) << " "
                << std::get<3>(row) << " "
                << std::get<4>(row) << " "
                << std::get<5>(row) << "\n";
    }

    outFile2.close();


	//write seedsize_B
	filename="seedsize_new.txt";
	filePath=myfolder+"/"+filename;
	
	std::ofstream outFile3(filePath);
	if (!outFile3) {
    	std::cerr << "Error opening file: " << filePath << std::endl;
	}
    
    std::vector<std::tuple<std::string, int, int, int, int, int>> data_seedsize;
    data_seedsize.push_back(make_tuple("B",10,20,30,40,50));
    for(int i=0;i<7;i++){
    	data_seedsize.push_back( make_tuple(vec_res[i][0].name,vec_res[i][0].seedsize,vec_res[i][1].seedsize,vec_res[i][2].seedsize,vec_res[i][3].seedsize,vec_res[i][4].seedsize) );
	}
		
    outFile3 << std::fixed << std::setprecision(3);

    for (const auto& row : data_seedsize) {
        outFile3 << std::get<0>(row) << " "
                << std::get<1>(row) << " "
                << std::get<2>(row) << " "
                << std::get<3>(row) << " "
                << std::get<4>(row) << " "
                << std::get<5>(row) << "\n";
    }

    outFile3.close();
	
}


int main(){
	double epsilon=0.2;
	//const char *file="uselections\\uselections_network_heterogeneous.txt";
	//const char *file="obamacare\\obamacare_network_heterogeneous.txt";
	const char *file="brexit\\brexit_network_heterogeneous.txt";
	//const char *file="abortion\\abortion_network_heterogeneous.txt";
	//const char *file="fracking\\fracking_network_heterogeneous.txt";
	//const char *file="iphone_samsung\\iphone_samsung_network_heterogeneous.txt";
	//const char *file="Slashdot\\Slashdot.txt";
	//const char *file="Epinions\\Epinions.txt";
	
	
	std::string folder(file);
	size_t pos = folder.find('\\');
	std::string beforeSlash = folder.substr(0, pos);

	double gamma=0.01;
	
	if(beforeSlash=="Slashdot" or beforeSlash=="Epinions"){
		readGraph2(file,gamma); 
	} 
	else readGraph(file,gamma);
	
	menu(beforeSlash,epsilon);



    return 0;
}
