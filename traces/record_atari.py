import gym
import cv2
def key2action(k):
    adict = {119:2, 115:3}
    a = adict.get(k)
    if a is None:
        return 0
    else:
        return a


env = gym.make('Pong-v0')
s = env.reset()

all_keys = []
all_states = [s, ]
cv2.namedWindow("a")
cv2.imshow("a", s)
ch = cv2.waitKey()
all_keys.append(ch)
a = key2action(ch)
while True:
    print(ch, a)
    s, _, _, _ = env.step(a)
    cv2.imshow("a", s)
    ch = cv2.waitKey()
    all_keys.append(ch)
    a = key2action(ch)
    if ch==113: # 'q'
        break


    
cv2.destroyAllWindows()

