# # importing librarys
# import cv2
# import numpy as npy
# import face_recognition as face_rec
# # function
# def resize(img, size) :
#     width = int(img.shape[1]*size)
#     height = int(img.shape[0] * size)
#     dimension = (width, height)
#     return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)


# # img declaration
# joshik = face_rec.load_image_file('krishna.jpg')
# joshik = cv2.cvtColor(joshik, cv2.COLOR_BGR2RGB)
# joshik = resize(joshik, 0.50)
# joshik_test = face_rec.load_image_file('elonmusk.jpg')
# joshik_test = resize(joshik_test, 0.50)
# joshik_test = cv2.cvtColor(joshik_test, cv2.COLOR_BGR2RGB)

# # finding face location

# faceLocation_joshik = face_rec.face_locations(joshik)[0]
# encode_joshik = face_rec.face_encodings(joshik)[0]
# cv2.rectangle(joshik, (faceLocation_joshik[3], faceLocation_joshik[0]), (faceLocation_joshik[1], faceLocation_joshik[2]), (255, 0, 255), 3)


# faceLocation_joshiktest = face_rec.face_locations(joshik_test)[0]
# encode_joshiktest = face_rec.face_encodings(joshik_test)[0]
# cv2.rectangle(joshik_test, (faceLocation_joshik[3], faceLocation_joshik[0]), (faceLocation_joshik[1], faceLocation_joshik[2]), (255, 0, 255), 3)

# results = face_rec.compare_faces([encode_joshik], encode_joshiktest)
# print(results)
# cv2.putText(joshik_test, f'{results}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2 )

# cv2.imshow('main_img', joshik)
# cv2.imshow('test_img', joshik_test)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# importing libraries
import cv2
import numpy as np
import face_recognition as face_rec

# function
def resize(img, size):
    width = int(img.shape[1] * size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)

# img declaration
krishna = face_rec.load_image_file('krishna.jpg')
krishna = cv2.cvtColor(krishna, cv2.COLOR_BGR2RGB)
krishna = resize(krishna, 0.50)

krishna_test = face_rec.load_image_file('krishna_test.jpg')
krishna_test = resize(krishna_test, 0.50)
krishna_test = cv2.cvtColor(krishna_test, cv2.COLOR_BGR2RGB)

# finding face location
faceLocation_krishna = face_rec.face_locations(krishna)[0]
encode_krishna = face_rec.face_encodings(krishna)[0]
cv2.rectangle(krishna, (faceLocation_krishna[3], faceLocation_krishna[0]), (faceLocation_krishna[1], faceLocation_krishna[2]), (255, 0, 255), 3)

faceLocation_krishna_test = face_rec.face_locations(krishna_test)[0]
encode_krishna_test = face_rec.face_encodings(krishna_test)[0]
cv2.rectangle(krishna_test, (faceLocation_krishna[3], faceLocation_krishna[0]), (faceLocation_krishna[1], faceLocation_krishna[2]), (255, 0, 255), 3)

results = face_rec.compare_faces([encode_krishna], encode_krishna_test)
print(results)
cv2.putText(krishna_test, f'{results}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('main_img', krishna)
cv2.imshow('test_img', krishna_test)
cv2.waitKey(0)
cv2.destroyAllWindows()
