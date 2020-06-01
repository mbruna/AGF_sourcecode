/*
 * METROPOLIS HASTINGS (STATIONARY) SIMULATION
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

    double timestep_ratio;

    unsigned int nout = 100;
    unsigned int steps_per_out = 1000;
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
            ("potential",po::value < unsigned int > (&potential)->default_value(0), "potential, 0=quadratic, 1=sin");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }


    const double delta = timestep_ratio * epsilon;

    std::cout << "Doing for " << nout*steps_per_out << " steps per sample. delta = " << delta << std::endl;
    std::cout << "Parameters: Nr = " << N << ", eps_r = " << epsilon <<", Nb = " << N_obs <<  ", eps_b = " << epsilon_obs << std::endl;

    auto positive_modulo = [&](int i, int n) {
        return (i % n + n) % n;
    };

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

    auto rho0 = [&](const double x) {
        if (potential==0){
            double a = 1.2; double Cb =(1-a/12)/(1+a/6); double varep2 = N_obs*PI*epsilon_br*epsilon_br; double Ct = 1.0/(1.0 - varep2*Cb);
            return Ct*(1.0 - varep2*b0(x));
        }
        else if (potential==1) {
            double varep2 = N_obs*PI*epsilon_br*epsilon_br; double bmin = 0.8162872472920132; double Ct = 1.0/(1.0 - varep2*bmin);
            return Ct*(1.0 - varep2*b0(x));
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
    std::vector<unsigned long long int> hist(2*num_box,0);

    std::cerr << "Average points per cell " << (N*samples*(1.0+nout*steps_per_out))/num_box << ". Max allowed " << (double)ULLONG_MAX << std::endl;

    if ( (N*samples*(1.0+nout*steps_per_out))/num_box > 0.1*(ULLONG_MAX)){
        std::cerr << "******** Too many steps for storage available. Avorting *************" << std::endl;
        return 1;
    }

    typedef position_d<2> position;
    Symbol<position> p;

    Symbol<id> id_; //particle id
    Symbol<hp> hp_; //histogram position

    VectorSymbolic<double,2> vector;
    VectorSymbolic<double,1> vector1;
    Normal normal;
    Uniform uni;
    std::uniform_real_distribution<double> unif(0,1);
    std::uniform_int_distribution<int> uni_int(0,N-1);
    std::normal_distribution<double> normi(0,1);

    AccumulateWithinDistance<std::bit_or<bool> > any_blue(epsilon_br);
    AccumulateWithinDistance<std::bit_or<bool> > any_red(epsilon);
    AccumulateWithinDistance<std::plus<vdouble2> > sum(epsilon);


    time_t start, end;
    time(&start);

    int rej_T = 0;

    std::cout << "Starting " << samples << " samples:" << std::endl;

    int n_samples_complete = 0;
    #pragma omp parallel for shared (hist, n_samples_complete, rej_T)
    for (unsigned int sample=0;sample<samples;sample++) {

        std::vector<unsigned long long int> hist_sample(2*num_box,0);
        generator_type gen(sample*time(0));

        moving_type particles;
        obstacles_type obstacles;

        obstacles.init_neighbour_search(vdouble2(-L/2,-L/2), vdouble2(L/2,L/2), vbool2(true,true));
        particles.init_neighbour_search(vdouble2(-L/2,-L/2), vdouble2(L/2,L/2), vbool2(false,false));

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
        int i_cand, rej=0;
        vdouble2 x_cand;

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

        for (int k=0; k<N; ++k) {

            //generate candidate position
            r = unif(gen) - L/2.0;
            while (unif(gen)> rho0(r)){
                r = unif(gen) - L/2.0;
            }

            //create new particle
            typename moving_type::value_type p;

            overlap = true;
            while (overlap) {
                get<position>(p) = vdouble2(r, unif(gen) - L/2.0);
                overlap = false;

                // loop over obstacles
                for (auto tpl: euclidean_search(obstacles.get_query(), get<position>(p), epsilon_br)) {
                    overlap = true;
                    break;
                }

                if (!overlap){
                    // loop over previous particles
                    for (auto tpl: euclidean_search(particles.get_query(), get<position>(p), epsilon)) {
                        overlap = true;
                        break;
                    }
                }

            }
            particles.push_back(p);
        }


        //initialize histogram for obstacles and particles
        for (moving_type::reference i: particles) {
            const double x = get<position>(i)[0];
            get<hp>(i) = std::floor((x + L/2.0)*num_box/L);
            hist_sample[get<hp>(i)]++;
        }

        for (obstacles_type::reference o: obstacles) {
            const double x = get<position>(o)[0];
            const size_t histogram_index = std::floor((x + L/2.0)*num_box/L);
            hist_sample[num_box + histogram_index]++;
        }

        for (int no = 0; no < nout; no++) {

            for (int ts = 0; ts < steps_per_out; ts++) {

                // candidate particle to be moved
                i_cand = uni_int(gen);
                size_t id_cand = get<id>(particles[i_cand]);
                x_cand = get<position>(particles[i_cand]) + delta * vdouble2(normi(gen), normi(gen));

                overlap = false;
                //check if candidate move is outside box
                if ((x_cand[0] < -L/2.0) || (x_cand[0] >= L/2.0) || (x_cand[1] < -L/2.0) || (x_cand[1] >= L/2.0)) {
                    rej += 1;
                    overlap = true;
                }

                if (!overlap) {
                    // loop over all obstacles
                    for (auto tpl: euclidean_search(obstacles.get_query(), x_cand, epsilon_br)) {
                        overlap = true;
                        rej += 1;
                        break;
                    }
                }

                if (!overlap) {
                    // loop over all particles except i_cand
                    for (auto tpl: euclidean_search(particles.get_query(), x_cand, epsilon)) {

                        const moving_type::value_type &j = std::get<0>(tpl);
                        if (get<id>(j) != id_cand) {
                            overlap = true;
                            rej += 1;
                            break;
                        }
                    }
                }

                if (!overlap) {
                    // if no overlap, accept candidate particle
                    get<position>(particles[i_cand]) = x_cand;
                    get<hp>(particles[i_cand]) = positive_modulo(static_cast<int>(std::floor((x_cand[0] + L / 2.0) * num_box / L)), num_box);
                    particles.update_positions(particles.begin() + i_cand, particles.begin() + i_cand + 1);
                }

                for (moving_type::reference i: particles) {
                    hist_sample[get<hp>(i)]++;
                }
            }
        }

        for (int ii=0; ii<hist.size(); ++ii) {
            unsigned long long int &tmp = hist[ii];
            #pragma omp atomic
            tmp += hist_sample[ii];
        }

        #pragma omp atomic
        rej_T += rej;
        //std::cout <<"Acceptance rate of nout " << no << " is "<< 1.0-(double)rej/(double)(steps_per_out) << std::endl;

        #pragma omp atomic
        n_samples_complete++;

        if (n_samples_complete%100==0) {
            std::cout << "samples complete: " << double(n_samples_complete)/samples*100 << " percent" << std::endl;
        }

    }
    std::cout <<"Global acceptance rate is " << 1.0 - (double)rej_T/(double)(samples*(1.0 + nout*steps_per_out)) << std::endl;
    std::cout << std::endl;
    time(&end);

    std::cout << "Done in "<<difftime(end,start) << std::endl;

    //write to info file
    struct tm * timeinfo;
    timeinfo = localtime ( &end );
    info.open("results/info_mh_nf.dat", std::ofstream::out | std::ofstream::app);
    info << "Current local time and date: " << asctime (timeinfo) << std::endl;
    info << "Directory is " << output_name << ". No flux. " << std::endl;
    info << "Doing for " << nout*steps_per_out << " steps per sample. Samples = " << samples << ". delta = " << delta << std::endl;
    info << "Parameters: Nr = " << N << ", eps_r = " << epsilon <<", Nb = " << N_obs <<  ", eps_b = " << epsilon_obs << std::endl;
    info << "Acceptance rate is: " << 1.0 - (double)rej_T/(double)(samples*(1.0 + nout*steps_per_out)) << std::endl;
    info << "==================" << std::endl << std::endl;
    info.close();

    double scaleby = 1.0/(N*samples*(1.0 + nout*steps_per_out)*dh*L);

    file.open((output_name+"/hist.dat").c_str());
    for (int l=0; l<num_box; l++){
        file << hist[l]*scaleby << " ";
    }
    file << std::endl;

    scaleby = 1.0/(samples*N_obs*dh*L);

    for (int l=0; l<num_box; l++){
        file << hist[num_box + l]*scaleby << " ";
    }
    file << std::endl;
    file.close();
}
