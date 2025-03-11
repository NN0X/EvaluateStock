#include <iostream>

#include "../include/mlpack.hpp"

// data format:
// {
//     "SMA5" : x, // 5 day moving average
//     "SMA10" : x, // 10 day moving average
//     "SMA20" : x, // 20 day moving average
//     "SMA50" : x, // 50 day moving average
//     "SMA100" : x, // 100 day moving average
//     "SMA200" : x, // 200 day moving average
//     "EMA5" : x, // 5 day exponential moving average
//     "EMA10" : x, // 10 day exponential moving average
//     "EMA20" : x, // 20 day exponential moving average
//     "EMA50" : x, // 50 day exponential moving average
//     "EMA200" : x, // 200 day exponential moving average
//     "MACD12" : x, // MACD over 12 days
//     "MACD26" : x, // MACD over 26 days
//     "BollingerUpper" : x, // Bollinger Upper Band over 20 days with 2 standard deviations
//     "BollingerLower" : x, // Bollinger Lower Band over 20 days with 2 standard deviations
//     "RSI14" : x, // Relative Strength Index over 14 days
//     "RSI28" : x, // Relative Strength Index over 28 days
//     "Stochastic14" : x, // Stochastic Oscillator over 14 days (80 is overbought, 20 is oversold)
//     "Stochastic28" : x, // Stochastic Oscillator over 28 days (80 is overbought, 20 is oversold)
//     "ROC14" : x, // Rate of Change over 14 days
//     "ROC28" : x, // Rate of Change over 28 days
//     "ROC50" : x, // Rate of Change over 50 days
//     "ATR14" : x, // Average True Range over 14 days
//     "ATR28" : x, // Average True Range over 28 days
//     "STDDEV14" : x, // Standard Deviation over 14 days
//     "STDDEV28" : x, // Standard Deviation over 28 days
//     "STDDEV50" : x, // Standard Deviation over 50 days
//     "DAY" : x, // day of the week (0-6)
//     "MONTH" : x, // month of the year (0-11)
//     "lastCost" : x, // last closing cost
//     "volume5" : x, // 5 day volume average
//     "volume10" : x, // 10 day volume average
//     "volume20" : x, // 20 day volume average
//     "volume50" : x, // 50 day volume average
// }

void trainModel(const std::string &modelPath, const std::string &dataPath)
{
        // Random Forest
}

void testModel(const std::string &modelPath, const std::string &dataPath)
{
        // Random Forest
}

enum StockPrediction
{
        BUY,
        SELL,
        HOLD
};

int evaluateStock(const std::string &modelPath, const std::string &dataPath)
{
        return HOLD;
}

int main(int argc, char **argv)
{
        if (argc == 2 && std::string(argv[1]) == "-h")
        {
                std::cout << "Usage: " << argv[0] << " <modelPath> <dataPath>\n";
                std::cout << "\tmodelPath: path to the model file\n";
                std::cout << "\tdataPath: path to the data file\n";
                std::cout << "\n\t\tUse -t to train the model\n";
                std::cout << "\n\t\tUse -e to evaluate the stock\n";
                std::cout << "\n\t\tUse -c to test the model\n";
                std::cout << "\n\t\tUse -h to display this help\n";
                return 0;
        }
        else if (argc < 3)
        {
                std::cerr << "Usage: " << argv[0] << " <modelPath> <dataPath>\n";
                std::cerr << "\tmodelPath: path to the model file\n";
                std::cerr << "\tdataPath: path to the data file\n";
                std::cerr << "\n\tUse -h for help\n";
                return 1;
        }

        if (argc == 4 && std::string(argv[1]) == "-t")
        {
                trainModel(argv[2], argv[3]);
        }
        else if (argc == 4 && std::string(argv[1]) == "-e")
        {
                evaluateStock(argv[2], argv[3]);
        }
        else if (argc == 4 && std::string(argv[1]) == "-c")
        {
                testModel(argv[2], argv[3]);
        }
        else if (argc == 3)
        {
                evaluateStock(argv[1], argv[2]);
        }
        else
        {
                std::cerr << "Invalid arguments\n";
                return 1;
        }

        return 0;
}
