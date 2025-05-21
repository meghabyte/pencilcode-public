
import re
import os
import urllib.parse

SAMPLE = "speed+Infinity|moveto+0,+300|pen+black,+10|moveto+100,+300|moveto+125,+275|moveto+100,+250|moveto+0,+250|moveto+-25,+275|moveto+0,+300|pen+black,+5|moveto+25,+300|moveto+25,+325|dot+black,+10|pu()|moveto+50,+300|pen+black,+5|moveto+50,+325|dot+black,+10|pu()|moveto+75,+300|pen+black,+5|moveto+75,+325|dot+black,+10|pu()|moveto+25,+275|dot+black,+10|moveto+75,+275|dot+black,+10|moveto+25,+250|pen+black,+10|moveto+-50,+175|pen+black,+5|moveto+-100,+175|dot+black,+10|pu()|moveto+-50,+175|pen+black,+10|moveto+25,+75|moveto+75,+75|moveto+150,+175|moveto+75,+250|pu()|moveto+150,+175|pen+black,+5|%60%60|"
def convert(one_line):
    #if
    x = re.search("^if(.+)$", one_line)
    if(x):
        return ["if"+x.group(1).replace("+"," ")+":"]
    #else if
    x = re.search("^else[+]if(.+)$", one_line)
    if(x):
        return ["else if"+x.group(1).replace("+"," ")+":"]
    #while
    x = re.search("^while(.+)$", one_line)
    if(x):
        return ["while"+x.group(1).replace("+"," ")+":"]
    #return
    x = re.search("^return(.+)$", one_line)
    if(x):
        return ["return"+x.group(1).replace("+"," ")]
    #for
    x = re.search("^for[+][[](.+)[.][.](.+)[]]$", one_line)
    if(x):
        return ["for i in range("+x.group(1)+","+x.group(2)+"):"]
    #for range
    x = re.search("^for[+](.+)[+]in[+][[](.+)[.][.](.+)[]]$", one_line)
    if(x):
        return ["for "+x.group(1)+" in range("+x.group(2)+","+x.group(3)+"):"]
    #listen, TODO: how to handle this better? 
    x = re.search("^listen[+][(](.*)[)][+]->$", one_line)
    if(x):
        return ["listen("+x.group(1)+")"]
    #write
    x = re.search("^write[+](.+)$", one_line)
    if(x):
        return ["write("+x.group(1)+")"]
    #play
    x = re.search("^play[+](.+)$", one_line)
    if(x):
        return ["playnotes("+x.group(1)+")"]
    #tone
    x = re.search("^tone[+](.+)$", one_line)
    if(x):
        return ["playtone("+x.group(1)+")"]
    #silence
    x = re.search("^silence[+]()$", one_line)
    if(x):
        return ["silencesound("+x.group(1)+")"]
    #say
    x = re.search("^say[+](.+)$", one_line)
    if(x):
        return ["say("+x.group(1)+")"]
    #pen
    x = re.search("^pen[+](\w+),[+](\w+)$", one_line)
    if(x):
        return ["pencolor('"+x.group(1)+"')", "pensize("+x.group(2)+")"]
    #pencolor
    x = re.search("^pen[+](\w+)", one_line)
    if(x):
        return ["pencolor('"+x.group(1)+"')"]
    #dot
    x = re.search("^dot[+](\w+),[+](\w+)$", one_line)
    if(x):
        return ["dot('"+x.group(1)+"',"+x.group(2)+")"]
    #wear
    x = re.search("^wear[+](.+)$", one_line)
    if(x):
        return ["wear('"+x.group(1)+"')"]
    #speed
    x = re.search("^speed[+](\w+)$", one_line)
    if(x):
        return ["speed("+x.group(1)+")"]
    #pause
    x = re.search("^pause[+](\w+)$", one_line)
    if(x):
        return ["pause("+x.group(1)+")"]
    # jumpto
    x = re.search("^jumpto[+](.+),(.+)$", one_line)
    if(x):
        return ["penup()", "goto("+x.group(1)+","+x.group(2)+")", "pendown()"]
    # jumpxy
    x = re.search("^jumpxy[+](.+),(.+)$", one_line)
    if(x):
        return ["penup()", "movexy("+x.group(1)+","+x.group(2)+")", "pendown()"]
    # moveto
    x = re.search("^moveto[+](.+),(.+)$", one_line)
    if(x):
        return ["moveto("+x.group(1).replace("+","")+","+x.group(2).replace("+","")+")"]
    # movexy
    x = re.search("^movexy[+](.+),(.+)$", one_line)
    if(x):
        return ["movexy("+x.group(1)+","+x.group(2)+")"]
    # turnto
    x = re.search("^turnto[+](.+),(.+)$", one_line)
    if(x):
        return ["setheading("+x.group(1)+","+x.group(2)+")"]
    # rt
    x = re.search("^rt[+](.+)$", one_line)
    if(x):
        return ["right("+x.group(1)+")"]
    # rt arc
    x = re.search("^rt[+](.+),(\w+)$", one_line)
    if(x):
        return ["rightarc("+x.group(1)+","+x.group(2)+")"]
    # lt
    x = re.search("^lt[+](.+)$", one_line)
    if(x):
        return ["left("+x.group(1)+")"]
    # lt arc
    x = re.search("^lt[+](.+),(\w+)$", one_line)
    if(x):
        return ["leftarc("+x.group(1)+","+x.group(2)+")"]
    # fd
    x = re.search("^fd[+](.+)$", one_line)
    if(x):
        return ["forward("+x.group(1)+")"]
    # bd
    x = re.search("^bk[+](.+)$", one_line)
    if(x):
        return ["backward("+x.group(1)+")"]
    # fill
    x = re.search("^fill[+](\w+)$", one_line)
    if(x):
        return ["fillcolor("+x.group(1)+")"]
    # wear
    x = re.search("^wear[+](\w+)$", one_line)
    if(x):
        return ["setimage("+x.group(1)+")"]
    # img
    x = re.search("^img[+](\w+)$", one_line)
    if(x):
        return ["writeimage("+x.group(1)+")"]
    # grow
    x = re.search("^grow[+](\w+)$", one_line)
    if(x):
        return ["grow("+x.group(1)+")"]
    # button
    x = re.search("^button[+]'(\w+)',[+]->$", one_line)
    if(x):
        return ["def button("+x.group(1)+"):"]
    # hide
    x = re.search("^hide[+]()$", one_line)
    if(x):
        return ["hide()"]
    # show
    x = re.search("^show[+]()$", one_line)
    if(x):
        return ["show()"]
    # clear screen
    x = re.search("^cs()$", one_line)
    if(x):
        return ["clearScreen()"]
    # pen up
    x = re.search("^pu[(][)]$", one_line)
    if(x):
        return ["penup()"]
    # pen down
    x = re.search("^pd()$", one_line)
    if(x):
        return ["pendown[(][)]"]
    # drawon
    x = re.search("^drawon[+](\w+)$", one_line)
    if(x):
        return ["drawon("+x.group(1)+")"]
    # box
    x = re.search("^box[+](\w+),[+](\w+)$", one_line)
    if(x):
        return ["box("+x.group(1)+","+x.group(2)+")"]
    # debug
    x = re.search("^debug[+](\w+)$", one_line)
    if(x):
        return ["debug("+x.group(1)+")"]
    # type
    x = re.search("^type[+](\w+)$", one_line)
    if(x):
        return ["type("+x.group(1)+")"]
    # typebox
    x = re.search("^typebox[+](\w+)$", one_line)
    if(x):
        return ["typebox("+x.group(1)+")"]
    #typeline
    x = re.search("^typeline[+]()$", one_line)
    if(x):
        return ["typeline()"]
    # label
    x = re.search("^label[+](.+)$", one_line)
    if(x):
        return ["label("+x.group(1)+")"]
    #print(one_line)
    # catchall
    x = re.search("^(.+)[+]=[+](.+)$", one_line)
    if(x):
        return [""+x.group(1)+"("+x.group(2)+")"]
    x = re.search("^(.+)[+](.+)$", one_line)
    if(x):
        return [""+x.group(1)+"("+x.group(2)+")"]
    x = re.search("^(.+)[+](.+)[+](.+)$", one_line)
    if(x):
        return [""+x.group(1)+"("+x.group(2)+","+x.group(3)+")"]
    x = re.search("(\w+)", one_line)
    if(x):
        return [""+x.group(1)+"()"]
    #print(one_line)
    return [one_line.replace("+"," ")]

def full_convert(pc_line):
    python_lines = []
    for l in pc_line.split("|"):
        if(l and l[0] == "+"):
            #indentation
            prefix_count = len(l) - len(l.lstrip('+'))
            prefix = " "*prefix_count
            python_lines = python_lines + [prefix+z for z in convert(l.lstrip('+'))]
        elif("t." in l):
            #indentation
            prefix = "t."
            python_lines = python_lines + [prefix+z for z in convert(l.lstrip('t.'))]
        else:
            python_lines = python_lines + convert(l)
    program_lines = "\n".join(python_lines)
    return program_lines

def convert_data_files(train_data_f="trace_logs_std/0/000"):
    for f in os.listdir(train_data_f)[1:2]:
        for fn in os.listdir(train_data_f+"/"+f):
            for fna in os.listdir(train_data_f+"/"+f+"/"+fn):
                with open(train_data_f+"/"+f+"/"+fn+"/"+fna) as trace_f:
                    lines = trace_f.readlines()
                    for l in lines:
                        program = full_convert(urllib.parse.unquote(l.split(", DATE")[0]))

#full_convert(SAMPLE)