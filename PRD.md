main task is voice-over or full dubbing video

pipline for two ways of dubbing: voice-over or full;
1) video to wav with appropriate kHz rate;
2) division into 2 stems(voice, music and noice);
3) first stage segmentation using ffmpeg of approximatly 40 to 90 sec segments (recommended non-fixed size);
    Into segments;
    1) second segmentation with pyannote.audio and whisperx;
    2) using LLM model for intelligent segmentation (assign right speaker for word);
    3) using LLM to translate and adjust the audio length to the original, and make emotion tone or audio effect (input_llm=output_text_segments '''for more context''');
    4) extracting features from second audio segments in loop (age, gender(sex));
    5) voicing translated and adjusted emotional marked text with appropriate gender(sex), age;
    6) in full dubbing mode use lip sync for each second segs
    7) connecting all second audio to common audio;
4) connecting all first segmentation audio in common final audio;
5) adding final audio to video, in case voice-over dubbing make value of original audio  is about 30% of final audio, in case full dubbing connect audio with video without original audio;
6) get path to final dubbing video;

implementation;
1) use ffmpeg;
2) use demucs;
3) use ffmpeg for segmentation by noise and save segments in format "first_seg/001_0.00-12.34/segment.wav";
    Implementation Into segments;
    1) use pyannote.audio and whisperx for got word_segments;
    2) use LLM, then split audio, make output_audio_segments and output_text_segments;
    3) use LLM with appropriate prompt for that (emotion are defined from context(output_text_segments));
    4) use dict when extarct age and gender(audeering/wav2vec2-large-robust-24-ft-age-gender);
    5) use Fish tts for voicing with emo tags;
    6) use sync labs api lip sync for each second segs in full dubbing mode cause has length restrictions on platform 
    7) use ffmpeg for connecting second audio seg, do not cut the audio, overlay it on top of each other if speaker is different and if it is necessary and use fit_audio to increase and decrease audio length if necessary but no more than 10% of the length;
4) use ffmpeg for connecting first audio segs;
5) Use ffmpeg to add the final audio to the original audio video so that the volume of the original audio voices is about 30% of the volume of the final audio in over-voice-dubbing case, but in full voice dubbing there are no original voice audio;
6) return path text of target video;

#deploy...

@for_client.txt - это описание сервера для других клиентов чтобы они понимали друг друга, здесь должно быть ролное описание как использовать проект, чтобы можно было сделать клиентскую часть  на основе этого файла 

