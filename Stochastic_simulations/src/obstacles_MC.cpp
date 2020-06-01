/*
 * TIME DEPENDENT MONTE CARLO SIMULATION
 */

#include "time.h"
#include <random>
#include <algorithm>

#include "Aboria.h"
using namespace Aboria;

#include "boost/program_options.hpp"
namespace po = boost::program_options;

int main(int argc, char **argv) {

    /*
     * setup parameters
     */
    const double PI = boost::math::constants::pi<double>();

    double L = 1.0; //domain size
    double D = 1.0; //diffusion coefficient
    double epsilon = 0.01; //particle size/potential range
    unsigned int N = 100; //number of particles

    double epsilon_obs = 0.015; //obstacle size
    unsigned int N_obs = 500; // number of obstacles
    unsigned int potential = 0; //blues distribution or potential

    double epsilon_br = (epsilon + epsilon_obs)/2;

    int output_point_positions;
    double timestep_ratio, final_time;

    unsigned int nout, init;

    unsigned int samples = 100; // number of samples
    const int num_box = 35; //number of boxes for histogram (in each coordinate)

    typedef std::mt19937 generator_type;

    std::string output_name;
    std::ofstream file, info;
    char buffer[100];


    /*
     * setup user input
     */

    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("particles", po::value < unsigned int > (&N)->default_value(100), "number of particles per sample")
            ("obstacles", po::value < unsigned int > (&N_obs)->default_value(500), "number of obstacles per sample")
            ("nout", po::value < unsigned int > (&nout)->default_value(10), "number of output points")
            ("samples", po::value < unsigned int > (&samples)->default_value(100), "number of samples")
            ("epsilon", po::value<double>(&epsilon)->default_value(0.01), "diameter of particles")
            ("epsilon_obs", po::value<double>(&epsilon_obs)->default_value(0.015), "diameter of obstacles")
            ("dt", po::value<double>(&timestep_ratio)->default_value(0.33), "average move divided by epsilon")
            ("output-name", po::value<std::string>(&output_name)->default_value("output"), "output file basename")
            ("potential",po::value < unsigned int > (&potential)->default_value(0), "potential, 0=quadratic, 1=sin")
            ("init",po::value < unsigned int > (&init)->default_value(0), "=0 for uniform with no obstacle overlaps, =1 for uniform with obstacle overlaps")
            ("final-time", po::value<double>(&final_time)->default_value(0.05), "total simulation time")
            ("output-point-positions", po::value<int>(&output_point_positions)->default_value(0),
             "output point positions");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }


    const double mean_s = timestep_ratio * epsilon;
    double dt = std::pow(mean_s, 2) / (4 * D);
    const double timesteps = final_time / dt;
    unsigned int timesteps_per_out = std::round(timesteps / nout);
    timesteps_per_out = std::max(timesteps_per_out,1u); // in case the value in the previous step gave zero (trying to output more tiems than timesteps).
    std::cout << "simulating for " << timesteps_per_out * nout << " timesteps. dt = " << dt << std::endl;
    std::cout << "Parameters: Nr = " << N << ", eps_r = " << epsilon <<", Nb = " << N_obs <<  ", eps_b = " << epsilon_obs << std::endl;

    // set function to have a max at 1 for the Rejection sampling
    auto b0 = [&](const double x) {
        if (potential==0) {
            double a = 1.2, C = 1/(1+a/6); double b = 1-a/12;
            return C*(a*x*x + b);
        }
        else if (potential==1){
            return 1.2*(1.0 + 0.1*sin(20*x))*(x*x + 0.75)/1.2652825333067244;
            //last number is the constant to make the maximum be one.
        }

    };

    const double dh = L / num_box; //width of the histogram box

    /*
     * setup aboria variables and structures
     */

    ABORIA_VARIABLE(hp, int, "hp");
    typedef Particles<std::tuple<hp>,2> moving_type;
    typedef Particles<std::tuple<hp>,2> obstacles_type;

    //add one row to histogram to store hist of blue obstacles (to be stored in the last row)
    std::vector<unsigned long long int> hist((nout+2)*num_box,0);

    typedef position_d<2> position;
    Symbol<position> p;

    Symbol<id> id_; //particle id
    Symbol<hp> hp_; //histogram position

    VectorSymbolic<double,2> vector;
    VectorSymbolic<double,1> vector1;
    Normal normal;
    Uniform uni;
    std::uniform_real_distribution<double> unif(0,1);

    AccumulateWithinDistance<std::plus<vdouble2> > sum_rr(epsilon);
    AccumulateWithinDistance<std::plus<vdouble2> > sum_br(epsilon_br);

    /*
     * Diffusion step for particles
     */

    time_t start, end;
    time(&start);
    std::cout << "Starting " << samples << " samples:" << std::endl;

    int n_samples_complete = 0;
    #pragma omp parallel for shared (hist, n_samples_complete)
    for (unsigned int sample=0;sample<samples;sample++) {
        std::vector<unsigned long long int> hist_sample((nout+2)*num_box,0);
        generator_type gen(sample*time(0));

        moving_type particles;
        obstacles_type obstacles;

        obstacles.init_neighbour_search(vdouble2(-L/2,-L/2), vdouble2(L/2,L/2), vbool2(true,true));
        particles.init_neighbour_search(vdouble2(-L,-L), vdouble2(L,L), vbool2(false,false));

        particles.set_seed((samples + N*sample)*time(0)); //set seed, uncorrelated from sample one.
        obstacles.set_seed((samples + N_obs*sample)*time(0)); //set seed, uncorrelated from sample one.

        Label<0, moving_type> i(particles);
        Label<1, moving_type> j(particles);
        Label<2, obstacles_type> o(obstacles);

        // dx is a symbol representing the difference in positions of particle i and j, or particle i and obstacle o.
        auto dx = create_dx(i, j);
        auto dy = create_dx(i, o);

        // place obstacles according to distribution
        int l; double r;
        bool overlap;

        for (int k=0; k<N_obs; ++k) {
            // generate candidate position
            r = unif(gen) - L/2.0;
            while (unif(gen) > b0(r)) {
                r = unif(gen) - L/2.0;
            }

            // create new obstacle
            typename obstacles_type::value_type p;

            overlap = true;
            while (overlap) {
                get<position>(p) = vdouble2(r, unif(gen) - L/2.0);
                overlap = false;

                // loop over previous obstacles
                for (auto tpl: euclidean_search(obstacles.get_query(), get<position>(p), epsilon_obs)) {
                    overlap = true;
                    break;
                }
            }
            obstacles.push_back(p);

        }

        // place particles, check for overlaps with all obstacles and other particles
        // init = 0: create uniform distribution without overlapping obstacles or previously placed particles
        // init = 1: create uniform distribution by placing particles only checking against particles (to make it more uniform, and then correct for overlaps)

        for (int k=0; k<N; ++k) {

            //create new particle
            typename moving_type::value_type p;

            overlap = true;
            while (overlap) {
                get<position>(p) = vdouble2(unif(gen) - L/2.0, unif(gen) - L/2.0);
                overlap = false;

                // loop over previous particles
                for (auto tpl: euclidean_search(particles.get_query(), get<position>(p), epsilon)) {
                    overlap = true;
                    break;
                }

                if ((!overlap)&&(init==0)) {

                    // loop over obstacles
                    for (auto tpl: euclidean_search(obstacles.get_query(), get<position>(p), epsilon_br)) {
                        overlap = true;
                        break;
                    }
                }

            }
            particles.push_back(p);
        }
        if (init==1){ // collisions with obstacles
            p[i] += sum_br(o, -2.0*(epsilon_br/norm(dy) - 1.0)*dy);

            //particles could have gone outside
            p[i] = vector(if_else(p[i][0] < -L/2, -L-p[i][0], p[i][0]), if_else(p[i][1] < -L/2, -L-p[i][1], p[i][1]) );
            p[i] = vector(if_else(p[i][0] >  L/2,  L-p[i][0], p[i][0]), if_else(p[i][1] >  L/2,  L-p[i][1], p[i][1]) );
        }


        //initial histogram for obstacles and particles
        for (moving_type::reference i: particles) {
            const double x = get<position>(i)[0];
            get<hp>(i) = std::floor((x + L/2.0)*num_box/L);
            if (get<hp>(i)<0) get<hp>(i)=0;
            else if  (get<hp>(i)>num_box-1) get<hp>(i)=num_box-1;
            hist_sample[get<hp>(i)]++;
        }

        for (obstacles_type::reference o: obstacles) {
            const double x = get<position>(o)[0];
            get<hp>(o) = std::floor((x + L/2.0)*num_box/L);
            hist_sample[num_box*(nout+1) + get<hp>(o)]++;
        }

        if (output_point_positions&&sample==0) {
#ifdef HAVE_VTK
            sprintf(buffer, "%s/obstacles", output_name.c_str());
            vtkWriteGrid(buffer, 0, obstacles.get_grid(true));
#endif
        }

        for (int no = 0; no < nout; no++) {
            if (output_point_positions&&sample==0) {
#ifdef HAVE_VTK
                sprintf(buffer, "%s/particles", output_name.c_str());
                vtkWriteGrid(buffer, no, particles.get_grid(true));
#endif
            }

            for (int ts = 0; ts < timesteps_per_out; ++ts) {

                p[i] += (//Brownian step
                        std::sqrt(2 * D * dt) * vector(normal[i],normal[i]));

                //hard sphere collisions
                p[i] += sum_rr(j, if_else(id_[j] != id_[i], -(epsilon/norm(dx)-1),0.0)*dx);

                p[i] += sum_br(o, -2.0*(epsilon_br/norm(dy) - 1.0)*dy);

                p[i] = vector(if_else(p[i][0] < -L/2, -L-p[i][0], p[i][0]), if_else(p[i][1] < -L/2, -L-p[i][1], p[i][1]) );
                p[i] = vector(if_else(p[i][0] > L/2, L-p[i][0], p[i][0]), if_else(p[i][1] > L/2, L-p[i][1], p[i][1]) );
            }

            //update histogram
            for (moving_type::reference i: particles) {
                const double x = get<position>(i)[0];
                get<hp>(i) = std::floor((x + L/2.0)*num_box/L);
                if (get<hp>(i)<0) {get<hp>(i)=0;}
                else if  (get<hp>(i)>num_box-1) { get<hp>(i)=num_box-1; }
                hist_sample[(no+1)*num_box + get<hp>(i)]++;
            }
        }

        for (int ii=0; ii<hist.size(); ++ii) {
            unsigned long long int &tmp = hist[ii];
            #pragma omp atomic
            tmp += hist_sample[ii];
        }

        #pragma omp atomic
        n_samples_complete++;

        if (n_samples_complete%100==0) {
            std::cout << "samples complete: " << double(n_samples_complete)/samples*100 << " percent" << std::endl;
        }
    }
    time(&end);
    std::cout <<"Done in "<<difftime(end,start) << std::endl;

    //write to info file
    struct tm * timeinfo;
    timeinfo = localtime ( &end );
    info.open("results/info_nf.dat", std::ofstream::out | std::ofstream::app);
    info << "Current local time and date: " << asctime (timeinfo) << std::endl;
    info << "Directory is " << output_name << ". No flux. " << std::endl;
    info << "Samples = " << samples << ". dt = " << dt << ". Tf = " << final_time << std::endl;
    info << "Parameters: Nr = " << N << ", eps_r = " << epsilon <<", Nb = " << N_obs <<  ", eps_b = " << epsilon_obs << std::endl;
    info << "==================" << std::endl << std::endl;
    info.close();


    double scaleby = 1.0/(samples*N*dh*L);

    file.open((output_name+"/hist.dat").c_str());
    for (int i=0; i<nout+1; i++){
        file << i*timesteps_per_out*dt << " ";
        for (int l=0; l<num_box; l++){
            file << hist[i*num_box +l]*scaleby << " ";
        }
        file << std::endl;
    }

    scaleby = 1.0/(samples*N_obs*dh*L);
    file << -1 << " ";
    for (int l=0; l<num_box; l++){
        file << hist[(nout+1)*num_box+l]*scaleby << " ";
    }
    file << std::endl;
    file.close();
}
