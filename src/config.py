# RAVDESS native sample rate is 48k
sample_rate = 48000

# path to data for glob
data_path = '/scratch/work/huangg5/ravdess_ser/data/audio_speech/Actor_*/*.wav'
# 'RAVDESS dataset/Actor_*/*.wav'

# shift emotions left to be 0 indexed for PyTorch
emotions_dict ={
    '0':'surprised',
    '1':'neutral',
    '2':'calm',
    '3':'happy',
    '4':'sad',
    '5':'angry',
    '6':'fearful',
    '7':'disgust'
}

# Additional attributes from RAVDESS to play with
emotion_attributes = {
    '01': 'normal',
    '02': 'strong'
}