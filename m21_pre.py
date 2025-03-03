from music21 import *

# 楽譜となるオブジェクト
score = stream.Score()

# 音符を格納する配列
notes = []

# C4の四分音符を生成
n1 = note.Note("C4")
notes.append(n1)

# 四分休符を生成
n2 = note.Rest()
notes.append(n2)

# 楽譜に音符情報を追加
score.append(notes)

# 楽譜の表示
score.show()
