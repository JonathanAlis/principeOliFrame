import numpy as np
import matplotlib.pyplot as plt
from moviepy import *
import librosa
from pathlib import Path
#import IPython.display as ipd
import numpy as np
from scipy.signal import medfilt
#import simpleaudio as sa
import pandas as pd
import crepe
import mido
from PIL import Image, ImageOps


def video_note_split(vName, threshold=0.8, tune_thresh=0.3, dur_thresh=0.1,
                     find_eyes=True, show_notes=False):
    """
    vName: string ending with .mp4 that contains the video with the notes
    creates a csv file containing the starting and ending of all notes
    """

    video = VideoFileClip(vName)
    audio = video.audio
    h, w = video.size
    fps = video.fps
    print('video size ', w, 'x', h, ', at ', fps, ' fps.')
    t = audio.duration
    sr = audio.fps
    print('audio duration: ', t, ', sampling rate: ', sr)
    print(audio)
    y = audio.to_soundarray(fps=sr)

    y = librosa.core.to_mono(np.transpose(y))

    time, frequency, confidence, activation = crepe.predict(y, sr, viterbi=True)
    mfr = medfilt(frequency, 21)
    midi = 69 + 12 * np.log2(mfr / 440)
    mcon = medfilt(confidence, 21)

    if show_notes:
        plt.plot(time, midi)
        plt.plot(time, np.min(midi) + mcon * (np.max(midi) - np.min(midi)))
        plt.show()

    registeringNote = False
    notas = {
        'midi': [],
        'inicio': [],
        'fim': [],
        'duracao': [],
        'i': [],
        'j': [],
        'avg conf': [],
        'std dev': [],
        'median freq': [],   # Hz
        'avg midi': []       # valor médio contínuo em MIDI
    }
    notas = pd.DataFrame(notas)

    for i in range(time.shape[0]):
        if confidence[i] > threshold:
            if not registeringNote:
                registeringNote = True
                row = pd.DataFrame({'midi': [midi[i]], 'inicio': [time[i]], 'i': [i]})
                notas.loc[len(notas)] = row.iloc[0]
        else:
            if registeringNote:
                registeringNote = False
                notas.at[notas.index[-1], 'j'] = i
                notas.at[notas.index[-1], 'fim'] = time[i]
                notas.at[notas.index[-1], 'duracao'] = (
                    notas.at[notas.index[-1], 'fim'] -
                    notas.at[notas.index[-1], 'inicio']
                )

                i0 = int(notas.at[notas.index[-1], 'i'])
                j0 = i
                notas.at[notas.index[-1], 'avg conf'] = np.average(confidence[i0:j0])
                notas.at[notas.index[-1], 'median freq'] = np.median(frequency[i0:j0])
                notas.at[notas.index[-1], 'avg midi'] = np.average(midi[i0:j0])

    # remove notas curtas demais
    indexLowDur = notas[notas['duracao'] < dur_thresh].index
    notas = notas.drop(indexLowDur, inplace=False)

    # remove notas desafinadas
    for index, row in notas.iterrows():
        i = int(notas.at[index, 'i'])
        j = int(notas.at[index, 'j'])
        note = int(round(np.average(midi[i:j])))
        std = np.sqrt(np.average(abs(midi[i:j] - note) ** 2))
        notesThatPassed = np.abs(midi[i:j] - note) < tune_thresh
        if not all(notesThatPassed):
            notas = notas.drop(index, inplace=False)
        else:
            notas.at[index, 'midi'] = note
            notas.at[index, 'std dev'] = std

    print("identified notes: ")
    print(notas)

    vName = vName.split('.')[0]
    notas.to_csv(vName + '.csv', index=False)
    return notas


def folder_note_split(instrument_name ,threshold=0.8, tune_thresh=0.3,dur_thresh=0.1):
    """
    folder: string containig a folder that contains .mp4 files
    """
    folder=Path('instruments')/Path(instrument_name)

    videos=[video_path for video_path in folder.iterdir() if video_path.exists() and video_path.suffix=='.mp4']
    print(videos)
    df_list=[]
    for vid in videos:
        dfcsv = video_note_split(str(vid), threshold, tune_thresh, dur_thresh)
        df_list.append(dfcsv)
    df=pd.concat(df_list, keys=videos,names=[folder, 'Notes'])
    df = df.sort_values(by='midi')

    df.to_csv(Path(folder)/Path(instrument_name+'.csv'))
    df=df.reset_index(level=[folder])
    return df



def midi_to_dict(midi_path, channel=None):
    """
    Lê um arquivo MIDI e retorna dict com listas:
    {
      "midi": [...],
      "start": [...],
      "end": [...],
      "velocity": [...],
      "channel": [...]
    }
    
    Args:
        midi_path (str): Caminho do arquivo MIDI.
        channel (int or None): Canal MIDI a filtrar (0-15). 
                               Se None, pega todos os canais.
    """
    mid = mido.MidiFile(midi_path)
    
    notes = []
    ongoing_notes = {}  # chave = (nota, canal), valor = (start_time, velocity)
    time = 0.0
    
    for msg in mid:  # percorre eventos já mesclados
        time += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            if channel is None or msg.channel == channel:
                ongoing_notes[(msg.note, msg.channel)] = (time, msg.velocity)
        elif msg.type in ("note_off", "note_on") and msg.velocity == 0:
            if channel is None or msg.channel == channel:
                key = (msg.note, msg.channel)
                if key in ongoing_notes:
                    start_time, vel = ongoing_notes.pop(key)
                    notes.append({
                        "midi": msg.note,
                        "start": start_time,
                        "end": time,
                        "velocity": vel,
                        "channel": msg.channel
                    })
    
    # organiza como dict de listas
    result = {
        "midi": [],
        "start": [],
        "end": [],
        "velocity": [],
        "channel": []
    }
    for n in sorted(notes, key=lambda x: x["start"]):
        for k in result.keys():
            result[k].append(n[k])
    
    return result

import os
import random

def get_random_image_from_path(path):
    """
    Associa um caminho de imagem a uma variável.
    Se o caminho for uma pasta, escolhe uma imagem aleatória dela.
    
    Args:
        path (str): O caminho para o arquivo de imagem ou pasta de imagens.
        
    Returns:
        str: O caminho completo para a imagem, ou None se nenhuma imagem for encontrada.
    """
    # Lista de extensões de imagem comuns para verificar
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

    # Verifica se o caminho existe
    if not os.path.exists(path):
        print(f"Erro: O caminho '{path}' não existe.")
        return None
        
    # Se o caminho for um arquivo
    if os.path.isfile(path):
        # Checa se o arquivo é uma imagem
        if path.lower().endswith(IMAGE_EXTENSIONS):
            print(f"'{path}' é um arquivo de imagem.")
            return path
        else:
            print(f"Erro: '{path}' não é um arquivo de imagem.")
            return None
            
    # Se o caminho for uma pasta
    elif os.path.isdir(path):
        print(f"'{path}' é uma pasta. Procurando por imagens...")
        image_files = []
        for root, _, files in os.walk(path):
            for file in files:
                # Checa a extensão de cada arquivo para ver se é uma imagem
                if file.lower().endswith(IMAGE_EXTENSIONS):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            print(f"Erro: A pasta '{path}' não contém arquivos de imagem.")
            return None
        else:
            # Escolhe uma imagem aleatória da lista
            random_image = random.choice(image_files)
            print(f"Imagem aleatória escolhida: {random_image}")
            return random_image
            
    else:
        print(f"Erro: O caminho '{path}' não é um arquivo nem uma pasta válida.")
        return None
    

def autotune_clip(clip: VideoFileClip, source_midi: float, target_midi: float):
    """
    Ajusta o pitch do vídeo para a nota alvo (em MIDI), sempre mantendo
    no intervalo de ±6 semitons (mesma oitava ou a mais próxima).

    Retorna o novo clip com pitch alterado (áudio e vídeo juntos).
    """
    semitone_shift = target_midi - source_midi

    # Traz para o intervalo [-6, 6]
    while semitone_shift > 6:
        semitone_shift -= 12
    while semitone_shift < -6:
        semitone_shift += 12

    # Converte para fator de velocidade
    factor = 2 ** (semitone_shift / 12)

    # Aplica pitch shift (via alteração de velocidade)
    new_clip = clip.with_speed_scaled(factor)

    return new_clip

def load_image_corrected(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # respeita orientação EXIF
    return np.array(img)

def create_clip(instrumento, midi_track, save_name, dur_mult=1,
                imgshape='vertical', autotune=True, fade_duration=0.05,
                pause_path=None):
    """
    Cria vídeo sequenciado a partir de notas.
    
    pause_path: caminho de uma imagem para preencher os tempos sem notas
    """
    # Carrega notas do instrumento
    notas = pd.read_csv(Path('instruments')/Path(instrumento)/(instrumento+'.csv'))
    instrument_col = [col for col in notas.columns if 'instruments' in col][0]

    if imgshape == 'vertical':
        target_aspect = 9 / 16
    elif imgshape == 'horizontal':
        target_aspect = 16 / 9
    else:
        target_aspect = 1
    closest_to_target = np.inf
    
    source_vids = {}
    for _, rows in notas.iterrows():
        if rows[instrument_col] not in source_vids:
            vid = VideoFileClip(rows[instrument_col])
            width, height = vid.w, vid.h
            source_vids[rows[instrument_col]] = vid
            # video com aspect ratio mais proximo do alvo
            if ((width/height) - float(target_aspect))**2 < closest_to_target:
                closest_to_target = ((width/height) - target_aspect)**2
                size = (width, height)
    print(size)
    if size[0]/size[1] > target_aspect:
        size = (int(size[1]*target_aspect), size[1])
    else:
        size = (size[0], int(size[0]/target_aspect))
    print("Target size:", size)
        
    print("Videos:", source_vids.keys())
    
    clips = []
    intervals = []  # armazenar (start, end) de cada nota
    
    for i in range(len(midi_track['midi'])):
        midi_note = midi_track['midi'][i]
        freq = librosa.midi_to_hz(midi_note)

        note_dur = midi_track['end'][i] - midi_track['start'][i]
        this_note = notas.loc[notas['midi'] == midi_note]

        if len(this_note) == 0:
            print(f"missing note of midi {midi_note}, trying closest with dur >= {note_dur}")
            this_note=notas.loc[notas['duracao'] >= note_dur].copy()
            this_note['diff'] = np.abs(this_note['midi'] - midi_track['midi'][i])
            notes_sorted = this_note.sort_values(by='diff')
            this_note = notes_sorted.head(3)

        with_eyes = this_note.dropna()
        if len(with_eyes) == 0:
            continue
        if not with_eyes.empty:
            this_note = with_eyes.sample(n=1)
        else:
            this_note = this_note.sample(n=1)

        this_note = this_note.iloc[0].to_dict()
        source_start_time = this_note['inicio']*dur_mult
        start_time = midi_track['start'][i]

        source = this_note[f'instruments/{instrumento}']
        note_clip = (source_vids[source]
                     .subclipped(source_start_time, source_start_time + note_dur)
                     .cropped(width=size[0], height=size[1], x_center=source_vids[source].w//2, y_center=source_vids[source].h//2)
                     .with_start(start_time)
                     .with_effects([afx.AudioFadeIn(fade_duration),
                                    afx.AudioFadeOut(fade_duration)]))
        
        if autotune:
            note_clip = autotune_clip(
                note_clip,
                target_midi=midi_note,
                source_midi=librosa.hz_to_midi(this_note['median freq'])
            )
        
        clips.append(note_clip)
        intervals.append((start_time, start_time + note_dur))

    # Adiciona pausas se o usuário forneceu imagem
    image_path = pause_path
    if image_path is not None and len(intervals) > 0:
        intervals = sorted(intervals, key=lambda x: x[0])
        all_start = intervals[0][0]
        all_end = max(end for _, end in intervals)

        # 🔹 Caso especial: se começa com espaço antes do primeiro clip
        if all_start > 0:
            orig_clip = ImageClip(load_image_corrected(image_path))
            gap_clip = (ImageClip(load_image_corrected(image_path))
                        .with_duration(all_start)   # do 0 até o primeiro início
                        .with_start(0)
                        .cropped(width=size[0], height=size[1], x_center=orig_clip.w//2, y_center=orig_clip.h//2)
                        .resized(size))
            clips.append(gap_clip)

        current_time = all_start
        for (start, end) in intervals:
            if start > current_time:  # gap encontrado
                gap_clip = (ImageClip(load_image_corrected(image_path))
                            .with_duration(start - current_time)
                            .with_start(current_time)
                            .cropped(width=size[0], height=size[1], x_center=orig_clip.w//2, y_center=orig_clip.h//2)
                            .resized(size))
                clips.append(gap_clip)
            current_time = max(current_time, end)

        # Se sobrar espaço no final
        if current_time < all_end:
            gap_clip = (ImageClip(load_image_corrected(image_path))
                        .with_duration(all_end - current_time)
                        .with_start(current_time)
                        .cropped(width=size[0], height=size[1], x_center=orig_clip.w//2, y_center=orig_clip.h//2)
                        .resized(size))
            clips.append(gap_clip)


    print("Renderizando...")
    cc = CompositeVideoClip(clips, size=size)
    cc.write_videofile(save_name)

    return


def substituir_audio(video_path: str, audio_path: str, output_path: str,
                     codec: str = "libx264", audio_codec: str = "aac", fps: int = 30):
    """
    Substitui o áudio de um vídeo por um novo arquivo de áudio.

    Args:
        video_path (str): caminho do vídeo original
        audio_path (str): caminho do áudio processado
        output_path (str): caminho do vídeo de saída
        codec (str): codec de vídeo (padrão: 'libx264')
        audio_codec (str): codec de áudio (padrão: 'aac')
        fps (int): frames por segundo do vídeo de saída
    """
    # Carregar vídeo
    video = VideoFileClip(video_path)

    # Carregar áudio processado
    novo_audio = AudioFileClip(audio_path)

    # Substituir áudio no vídeo
    video_final = video.set_audio(novo_audio)

    # Exportar resultado
    video_final.write_videofile(output_path, codec=codec, audio_codec=audio_codec, fps=fps)



def create_clip_unrestricted(instrumento, midi_track, save_name, dur_mult=1,
                imgshape='vertical', autotune=True, fade_duration=0.05,
                pause_path=None):
    """
    Cria vídeo sequenciado a partir de notas.
    
    pause_path: caminho de uma imagem para preencher os tempos sem notas
    """
    # Carrega notas do instrumento
    notas = pd.read_csv(Path('instruments')/Path(instrumento)/(instrumento+'.csv'))
    instrument_col = [col for col in notas.columns if 'instruments' in col][0]

    if imgshape == 'vertical':
        target_aspect = 9 / 16
    elif imgshape == 'horizontal':
        target_aspect = 16 / 9
    else:
        target_aspect = 1
    closest_to_target = np.inf
    
    source_vids = {}
    for _, rows in notas.iterrows():
        if rows[instrument_col] not in source_vids:
            vid = VideoFileClip(rows[instrument_col])
            width, height = vid.w, vid.h
            source_vids[rows[instrument_col]] = vid
            # video com aspect ratio mais proximo do alvo
            if ((width/height) - float(target_aspect))**2 < closest_to_target:
                closest_to_target = ((width/height) - target_aspect)**2
                size = (width, height)
    print(size)
    if size[0]/size[1] > target_aspect:
        size = (int(size[1]*target_aspect), size[1])
    else:
        size = (size[0], int(size[0]/target_aspect))
    print("Target size:", size)
        
    print("Videos:", source_vids.keys())
    
    clips = []
    intervals = []  # armazenar (start, end) de cada nota
    
    for i in range(len(midi_track['midi'])):
        midi_note = midi_track['midi'][i]
        freq = librosa.midi_to_hz(midi_note)

        note_dur = midi_track['end'][i] - midi_track['start'][i]
        while note_dur > 0.0:
            this_note = notas.loc[notas['midi'] == midi_note]

            if len(this_note) == 0:
                print(f"missing note of midi {midi_note}, trying closest with dur >= {note_dur}")
                this_note=notas.loc[notas['duracao'] >= note_dur].copy()
                this_note['diff'] = np.abs(this_note['midi'] - midi_track['midi'][i])
                notes_sorted = this_note.sort_values(by='diff')
                this_note = notes_sorted.head(3)

            with_eyes = this_note.dropna()
            if len(with_eyes) == 0:
                note_dur -= 0.05
                continue
            if not with_eyes.empty:
                this_note = with_eyes.sample(n=1)
            else:
                this_note = this_note.sample(n=1)

            this_note = this_note.iloc[0].to_dict()
            source_start_time = this_note['inicio']*dur_mult
            start_time = midi_track['start'][i]

            source = this_note[f'instruments/{instrumento}']
            note_clip = (source_vids[source]
                        .subclipped(source_start_time, source_start_time + note_dur)
                        .cropped(width=size[0], height=size[1], x_center=source_vids[source].w//2, y_center=source_vids[source].h//2)
                        .with_start(start_time)
                        .with_effects([afx.AudioFadeIn(fade_duration),
                                        afx.AudioFadeOut(fade_duration)]))
            
            if autotune:
                note_clip = autotune_clip(
                    note_clip,
                    target_midi=midi_note,
                    source_midi=librosa.hz_to_midi(this_note['median freq'])
                )
            
            clips.append(note_clip)
            intervals.append((start_time, start_time + note_dur))
            note_dur = 0.0  # sair do loop


    # Adiciona pausas se o usuário forneceu imagem
    image_path = pause_path
    if image_path is not None and len(intervals) > 0:
        intervals = sorted(intervals, key=lambda x: x[0])
        all_start = intervals[0][0]
        all_end = max(end for _, end in intervals)

        # 🔹 Caso especial: se começa com espaço antes do primeiro clip
        if all_start > 0:
            orig_clip = ImageClip(load_image_corrected(image_path))
            gap_clip = (ImageClip(load_image_corrected(image_path))
                        .with_duration(all_start)   # do 0 até o primeiro início
                        .with_start(0)
                        .cropped(width=size[0], height=size[1], x_center=orig_clip.w//2, y_center=orig_clip.h//2)
                        .resized(size))
            clips.append(gap_clip)

        current_time = all_start
        for (start, end) in intervals:
            if start > current_time:  # gap encontrado
                gap_clip = (ImageClip(load_image_corrected(image_path))
                            .with_duration(start - current_time)
                            .with_start(current_time)
                            .cropped(width=size[0], height=size[1], x_center=orig_clip.w//2, y_center=orig_clip.h//2)
                            .resized(size))
                clips.append(gap_clip)
            current_time = max(current_time, end)

        # Se sobrar espaço no final
        if current_time < all_end:
            gap_clip = (ImageClip(load_image_corrected(image_path))
                        .with_duration(all_end - current_time)
                        .with_start(current_time)
                        .cropped(width=size[0], height=size[1], x_center=orig_clip.w//2, y_center=orig_clip.h//2)
                        .resized(size))
            clips.append(gap_clip)


    print("Renderizando...")
    cc = CompositeVideoClip(clips, size=size)
    cc.write_videofile(save_name)

    return
