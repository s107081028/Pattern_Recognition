import torchaudio
effects = [
    ["loudness", "0.5"],
    ["speed", "2.0"],
]
waveform, sr = torchaudio.load("./data/audios/Wheeze.2022-08-18_004037_759-20220810100304__ACHEST_pos0_D_MODE_8KREC.raw_fil_wRj73Ax4.wav")
torchaudio.save('./origin.wav', waveform, sr)
waveform, sr = torchaudio.sox_effects.apply_effects_tensor(waveform, sr, effects)
torchaudio.save('./Twicespeed.wav', waveform, sr)