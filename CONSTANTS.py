########################################################
#                   NAS PARAMETERS                     #
########################################################
CONTROLLER_SAMPLING_EPOCHS = 14 # number of epoch for controller sampling
SAMPLES_PER_CONTROLLER_EPOCH = 10 #sample architecture per controller epoch
CONTROLLER_TRAINING_EPOCHS = 10
ARCHITECTURE_TRAINING_EPOCHS = 5
CONTROLLER_LOSS_ALPHA = 0.9

########################################################
#               CONTROLLER PARAMETERS                  #
########################################################
CONTROLLER_LSTM_DIM = 100
CONTROLLER_OPTIMIZER = 'Adam'
CONTROLLER_LEARNING_RATE = 0.01
CONTROLLER_DECAY = 0.1
CONTROLLER_MOMENTUM = 0.0
CONTROLLER_USE_PREDICTOR = False

########################################################
#                   MODEL PARAMETERS                     #
########################################################
MAX_ARCHITECTURE_LENGTH = 3
OPTIMIZER = 'rmsprop'
LEARNING_RATE = 0.01
#DECAY = 0.0         #ntuk mengurangi learning rate seiring berjalannya waktu, 
                        #yang dapat membantu proses pelatihan konvergen lebih baik
#MOMENTUM = 0.0      #mengontrol seberapa besar langkah-langkah 
                        #yang diambil berdasarkan histori iterasi sebelumny
DROPOUT = 0.2
LOSS_FUNCTION = 'categorical_crossentropy'
ONE_SHOT = True     #mengontrol apakah akan menggunakan one shot trainig atau tidak
                        #one shot training untuk mengambil pengetahuan atau bobot (weights)
                        # yang sudah ada dari model sebelumnya dan menggunakannya sebagai 
                        # inisialisasi atau titik awal untuk melatih model baru

########################################################
#                   DATA PARAMETERS                    #
########################################################
TARGET_CLASSES = 4
#1: fix
#2: saccade
#3: sp
#4: noise

########################################################
#                  OUTPUT PARAMETERS                   #
########################################################
TOP_N = 1
