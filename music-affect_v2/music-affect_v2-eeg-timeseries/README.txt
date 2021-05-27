Each xlsx file corresponds to EEG signals from F7 channel of all participants listening to one song 

Column A is the timeseries info

Column B onwards are participants data, each column represents a participant (P1,P2...,P24)

S1, S2, S3, S4 = classical; S5,S6, S7, S8 = instrumental; S9, S10, S11, S12 = pop

You can use the same label info as music-eeg-features.xlsx to do predictions. 

Please note that this is the raw band power data exported from Emotiv EPOC, it might have went through some default preprocessing done by device SDK. If you want to do further pre-processing on the data, you can try min-max normalization and median smoothing filtering.


Paper:

Rahman, J.S., Gedeon, T., Caldwell, S. and Jones, R., 2020, July. Brain Melody Informatics: Analysing Effects of Music on Brainwave Patterns. In 2020 International Joint Conference on Neural Networks (IJCNN) (pp. 1-8). IEEE.