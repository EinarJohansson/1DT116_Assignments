///////////////////////////////////////////////////
// Low Level Parallel Programming 2017.
//
// 
//
// The main starting point for the crowd simulation.
//



#undef max
#include "ped_model.h"
#include "ParseScenario.h"

#include <thread>

#include "Simulation.h"
#include "TimingSimulation.h"
#include "ExportSimulation.h"
#ifndef NOQT
#include "QTSimulation.h"
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QApplication>
#include <QTimer>
#include "MainWindow.h"
#endif
#include <iostream>
#include <chrono>
#include <ctime>
#include <cstring>

#pragma comment(lib, "libpedsim.lib")

#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>


void print_usage(char *command) {
    printf("Usage: %s [--timing-mode|--export-trace[=export_trace.bin]] [--max-steps=100] [--help] [--cuda|--simd|--omp|--pthread|--seq] [scenario filename]\n", command);
    printf("There are three modes of execution:\n");
#ifndef NOQT
    printf("\t the QT window mode (default if no argument is provided. But this is also deprecated. Please opt to use the --export-trace mode instead)\n");
#endif
    printf("\t the --export-trace mode: where the agent movement are stored in a trace file and can be visualized by a separate python tool.\n");
    printf("\t the --timing-mode: the mode where no visualization is done and can be used to measure the performance of your implementation/optimization\n");
    printf("\nIf you need visualization, please try using the --export-trace mode. You can even copy the trace file to your computer and locally run the python visualizer. (You'll need to fork the assignment repository on your local machine too.)\n");
}

int main(int argc, char*argv[]) {
    bool timing_mode = false;
#ifndef NOQT
    bool export_trace = false; // If no QT, export_trace is default
#else
    bool export_trace = true;
#endif
    std::string scenefile = std::string("scenario.xml");
    int max_steps = 1000;
    Ped::IMPLEMENTATION implementation_to_test = Ped::SEQ;
    std::string export_trace_file = "";

    // Parsing the command line arguments. Feel free to add your own
    // configurations.
    // You may use the commented lines below to understand how to use the
    // getopt tool.
    while (1) {
        static struct option long_options[] = {
            {"timing-mode", no_argument, NULL, 't'},
            {"export-trace", optional_argument, NULL, 'e'},
            {"max-steps", required_argument, NULL, 'm'},
            {"help", no_argument, NULL, 'h'},
            {"cuda", no_argument, NULL, 'c'},
            {"simd", no_argument, NULL, 's'},
            {"omp", no_argument, NULL, 'o'},
            {"pthread", no_argument, NULL, 'p'},
            {"seq", no_argument, NULL, 'q'},
            {0, 0, 0, 0}  // End of options
        };

        int option_index = 0;
        int long_opt = getopt_long(argc, argv, "", long_options, &option_index);

        if (long_opt == -1) {
            break;  // No more long options
        }

        switch (long_opt) {
            case 't':
                // Handle --timing-mode
                std::cout << "Option --timing-mode activated\n";
                timing_mode = true;
                export_trace = false;
                break;
            case 'e':
                // Handle --export-trace
                export_trace = true;
                if (optarg != NULL) {
                    // If an argument is provided, set it as the export filename
                    export_trace_file = optarg;
                } else {
                    // If no argument is provided, use a default filename
                    export_trace_file = "export_trace.bin";
                }
                std::cout << "Option --export-trace set to: " << export_trace_file << std::endl;
                break;
            case 'h':
                // Handle --help
                print_usage(argv[0]);
                exit(0);
                break;
            case 'c':
                // Handle --cuda
                std::cout << "Option --cuda activated\n";
                implementation_to_test = Ped::CUDA;
                break;
            case 's':
                // Handle --simd
                std::cout << "Option --simd activated\n";
                implementation_to_test = Ped::VECTOR;
                break;
            case 'o':
                // Handle --omp
                std::cout << "Option --omp activated\n";
                implementation_to_test = Ped::OMP;
                break;
            case 'p':
                // Handle --pthread
                std::cout << "Option --pthread activated\n";
                implementation_to_test = Ped::PTHREAD;
                break;
            case 'q':
                // Handle --seq
                std::cout << "Option --seq activated\n";
                implementation_to_test = Ped::SEQ;
                break;
            case 'm':
                // Handle --max-steps with a numerical argument
                max_steps = std::stoi(optarg);  // Convert the argument to an integer
                std::cout << "Option --max-steps set to: " << max_steps << std::endl;
                break;
            default:
                // Handle unknown long options
                print_usage(argv[0]);
                exit(1);
        }
    }

    // Check if there is a filename argument (after all options)
    if (optind < argc) {
        scenefile = argv[optind];  // Set scenefile to the provided filename
    }

    if (export_trace) {
        if (max_steps > 200) {
            std::cout << "Reducing the maximum number of steps to 200 for the tracing run." << std::endl;
            std::cout << "The trace files can become really large!" << std::endl;
            max_steps = 200;
        }
    }

    int retval = 0;
    { // This scope is for the purpose of removing false memory leak positives

        // Timing version
        // Run twice, without the gui, to compare the runtimes.
        if (timing_mode) {
            // Run sequentially
            double fps_seq, fps_target;
            {
                Ped::Model model;
                ParseScenario parser(scenefile);
                // TODO: Byt ut parser.getAgents mot våra vektorer.
                model.setup(parser.getAgents(), parser.getX(), parser.getY(), parser.getWaypoints(), Ped::SEQ);
                Simulation *simulation = new TimingSimulation(model, max_steps);

                // Simulation mode to use when profiling (without any GUI)
                std::cout << "Running reference version...\n";
                auto start = std::chrono::steady_clock::now();
                simulation->runSimulation();
                auto duration_seq = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
                fps_seq = ((float)simulation->getTickCount()) / ((float)duration_seq.count())*1000.0;
                cout << "Reference time: " << duration_seq.count() << " milliseconds, " << fps_seq << " Frames Per Second." << std::endl;

                delete simulation;
            }

            {
                Ped::Model model;
                ParseScenario parser(scenefile);
                model.setup(parser.getAgents(), parser.getX(), parser.getY(), parser.getWaypoints(), implementation_to_test);
                Simulation *simulation = new TimingSimulation(model, max_steps);
                // Simulation mode to use when profiling (without any GUI)
                std::cout << "Running target version...\n";
                auto start = std::chrono::steady_clock::now();
                simulation->runSimulation();
                auto duration_target = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
                fps_target = ((float)simulation->getTickCount()) / ((float)duration_target.count())*1000.0;
                cout << "Target time: " << duration_target.count() << " milliseconds, " << fps_target << " Frames Per Second." << std::endl;

                delete simulation;
            }
            std::cout << "\n\nSpeedup: " << fps_target / fps_seq << std::endl;
        } else if (export_trace) {
                Ped::Model model;
                ParseScenario parser(scenefile);
                model.setup(parser.getAgents(), parser.getX(), parser.getY(), parser.getWaypoints(), implementation_to_test);

                Simulation *simulation = new ExportSimulation(model, max_steps, export_trace_file);

                std::cout << "Running Export Tracer...\n";
                auto start = std::chrono::steady_clock::now();
                simulation->runSimulation();
                auto duration_target = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
                float fps = ((float)simulation->getTickCount()) / ((float)duration_target.count())*1000.0;
                cout << "Time: " << duration_target.count() << " milliseconds, " << fps << " Frames Per Second." << std::endl;

                delete simulation;
#ifndef NOQT
        } else {
            // Graphics version
            Ped::Model model;
            ParseScenario parser(scenefile);
            model.setup(parser.getAgents(), parser.getX(), parser.getY(), parser.getWaypoints(), implementation_to_test);

            QApplication app(argc, argv);
            MainWindow mainwindow(model);

            QTSimulation simulation(model, max_steps, &mainwindow);

            cout << "Demo setup complete, running ..." << endl;

            // Simulation mode to use when visualizing
            auto start = std::chrono::steady_clock::now();
            mainwindow.show();
            simulation.runSimulation();
            retval = app.exec();

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
            float fps = ((float)simulation.getTickCount()) / ((float)duration.count())*1000.0;
            cout << "Time: " << duration.count() << " milliseconds, " << fps << " Frames Per Second." << std::endl;

#endif
        }
    }

    cout << "Done" << endl;
    return retval;
}
