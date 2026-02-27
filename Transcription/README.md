# Clip Extraction Experiments for IntelUpsell

## Overview
This folder within the repository has methods and an associated notebook for the clip extraction phase of the overall IntelUpsell pipeline.

## Proposed Pipeline
Raw audio files (15 minutes in length) are passed through standard preprocessing steps (decode, bandpass, AGC, limiter).

15 minute raw audio files are split (with respect to frames to avoid audio loss) into 10 ~1.5 minute clips.

Each of these ~1.5 minute clips is transcribed via gpt-4o-transcribe API.
- gpt-4o-transcribe struggles with transcribing very long audio files. For ~1.5 minute long clips it does a great job

Each transcript is passed to gpt-4o API, and the model is asked to identify the number of transcations (individual customer-operator interactions) within the transcript.
- In addition to the number of transactions, gpt-4o will break the passed transcript into a list, with each entry corresponding to a unique transaction. These are referred to as "transaction-specific transcripts," as they are transcripts that contain only 1 transaction but could be part of/entire transaction.
- gpt-4o will also return a status, either "complete" or "incomplete" for each individually identified transaction. This helps identify whether a transaction-specific transcript is part of or the entire transaction.

Using stable-whisper's align functionality, timestamps for each transaction-specific transcript are obtained.
- This was the part of the process where I found lots of struggles. Details are below

For each transcation-specific transcript, we slice the processed audio at associated timestamps

Depending on the transaction-specific transcript status (either complete or incomplete), we merge processed audio clips together to form transaction clips
- If the clip from t_0 - t_1 is marked as incomplete, it will be merged with the following clip from t_1 - t_2. If t_1 - t_2 is marked as complete, then a new transaction will start, otherwise merging continues until a complete transcript is found.

## Succesful Portions of Pipeline and Findings
### Raw Audio Decoding
This is a standard process that was easy to implement and mirrored the process in our Voice ID experiments.
Relevant methods that are taken from hoptix repo:
- prepare_audio_for_embedding()
- decode_audio()
- preprocess_audio()
- save_audio()

### Processed Audio Splitting
We initially struggled with naively splitting the 15 minute audio into 1.5 minute chunks, leading to audio loss at the splits. Instead of splitting at timestamps, we use ffmpeg atrim which splits directly at frames of audio to avoid audio loss.
Relevant method:
- split_audio_equal_parts_gapless()

### Transcribing ~1.5 Minute Audio Chunks
This is a simple process using gpt-4o-transcribe API. A standard prompt was created to produce transcript

### Identifying Number of Transactions in Each Audio Transcript
10 transcripts are passed to gpt-4o API and LLM is tasked with finding number of transactions in each transcript and transaction status. Prompt engineering greatly helped the outputs here (refer to prompt_2 local variable for instrucitons and return format). The main return here are transcation-specific transcripts

### Merging Transaction-Specific Transcripts to Form Whole-Transaction Transcripts
In the proposed pipeline, this step needs to happen after timestamps for each transaction-specific transcript is obtained. However, in early experimentation this step happened first, hence why it is solved in the "Transcription Experiments" notebook.

Leveraging the status fields that are returned by gpt-4o, it is easy to merge transaction-specific transcripts into a transcript corresponding to an entire transcript, which are referred to as "whole-transaction" transcripts. By iterating through both transaction-specific transcripts and associated statuses, we can compile whole-transaction transcripts.

## Failures of Pipeline and Findings
### Timestamping Overview and Process
We attempted to use stable-whisper to find timestamps in each 1.5 minute audio chunk for transcaction-specific transcripts. If effective, it would be very easy to slice processed audio and get a precise audio portion corresponding to a transaction

For each 1.5 minute audio chunk, we produce a stable-whisper transcription. stable-whisper will inherently provide timestamps for words/phrases in transcription output.

Using a self-deployed matching algorithm, we match the gpt-4o-transcribe transaction-specific transcript to the stable-whisper transcript, finding start and end timestamps corresponding to the matched region
- We implement fuzzy matching in two ways: matching entire transcript to transcript and matching anchors in gpt-4o-transcribe transcript to stable-whisper transcript
- Relevant methods:
  - best_segment_window_full()
  - best_segment_window_anchors()

After a succesful match we obtain the match timestamps and split the 1.5 minute chunks to only contain the match region.

If necessary, align this refined audio with the gpt-4o-transcribe transcript for more precise timestamps


### Problems Found
stable-whisper has align functionality which we have previously used to align a gpt-4o-transcribe transcript with an audio file. Unfortunately, align struggles with partial alignment, which is our case. Each transaction-specific transcript will never encompass the full 1.5 minute audio chunk (they can be as short as 4 seconds and as long as a minute). This meant that to find the timestamps of our gpt-4o-transcribe transaction-specific transcripts, we could not directly use stable-whisper's align method.

stable-whisper also supports word matching and word finding after producing a transcript (which would help us grab timestamps of the matched region). I tested this functionality by trying to match the gpt-4o-transcribe transaction-specific transcript to the stable-whisper transcript. However, stable-whisper's word matching and word finding was very rigid, it would only be succesful if the gpt-4o-transcribe transaction-specific transcript was practically equal to the stable-whisper transcript. For this reason, the proposed pipeline resorts to fuzzy matching (two different methods were tested). 

Fuzzy matching also had a few problems in both the anchor and full segment approaches. There are instances where there are good cuts of the 1.5 minute audio, but more often than not there will be missing audio or unnecessary inclusions of empty audio. Additionally, there are instances where both fuzzy matching algorithms are unable to make a match between transaction-specific transcript and whisper transcript.

The key dependency in this process is the stable-whisper transcription. If that is not accurate, this process will be impossible. I believe that these transcriptions are not good enough. I've seen examples where sentences are present in audio but only a few words are picked up by the stable-whisper transcription.