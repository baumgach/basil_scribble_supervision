import telepot as tp
import argparse

parser = argparse.ArgumentParser(description='Check experiment for termination.')
parser.add_argument('time_since_edit', type=int, help="Time since experiment run last edited a file")
parser.add_argument('exp_name', type=str, help="Name of experiment")
parser.add_argument('terminating', type=int, help="Whether watchdog is terminating")

args = parser.parse_args()

#Prepare bot
CHAT_ID = 358052240
API_KEY = '394942955:AAGIznxaj8zNB2LE-KMRa-JdQ5iyf1P-EBc'
bot = tp.Bot(API_KEY)

message = "It has been {} minutes since experiment {} last edited a file.".format(args.time_since_edit, args.exp_name)
if args.terminating == 1:
    message+= "\nTerminating watchdog."

bot.sendMessage(CHAT_ID, message)


