from pathlib import Path
import json, math, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, FancyArrowPatch, PathPatch, Polygon
from matplotlib.path import Path as MPath
from matplotlib import colors as mc
from matplotlib import font_manager
import fitz
ROOT=Path(__import__('os').environ['FROST_FIGURE_WORK'])
OUT=ROOT/"outputs/frost_figures_compact"
TMP=ROOT/"tmp/pdfs/frost_redesign"
OUT.mkdir(parents=True,exist_ok=True)
FONT_DIR=TMP/"source_sans_fonts"
for font_file in FONT_DIR.iterdir():
    if font_file.suffix.lower()==".ttf":
        font_manager.fontManager.addfont(str(font_file))
font_manager.findfont(font_manager.FontProperties(family=__import__("os").environ["FROST_DIAGRAM_FONT_FAMILY"]),fallback_to_default=False)
plt.rcParams.update({"font.family":__import__("os").environ["FROST_DIAGRAM_FONT_FAMILY"],"font.size":7.2,"mathtext.fontset":"custom",
    "mathtext.rm":__import__("os").environ["FROST_DIAGRAM_FONT_FAMILY"],"mathtext.it":__import__("os").environ["FROST_DIAGRAM_FONT_FAMILY"]+":italic",
    "mathtext.bf":__import__("os").environ["FROST_DIAGRAM_FONT_FAMILY"]+":bold","mathtext.bfit":__import__("os").environ["FROST_DIAGRAM_FONT_FAMILY"]+":bold:italic",
    "mathtext.sf":__import__("os").environ["FROST_DIAGRAM_FONT_FAMILY"],"mathtext.cal":"cmsy10","mathtext.fallback":"stixsans",
    "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","axes.linewidth":.65,"savefig.facecolor":"white"})
C={"ink":"#17232C","muted":"#4D5962","line":"#55636E","rule":"#CBD3DA","pale":"#F4F6F7",
   "sc":"#4B83AF","fsm":"#D47860","tt":"#429F92","scalar":"#C09A4A","green":"#C09A4A"}
CAT=[C["sc"],C["fsm"],C["green"],C["tt"]]
W=510.2362
TEXTS=[]
def tint(c,white=.89):
    return tuple(np.array(mc.to_rgb(c))*(1-white)+white)
def edge(c):
    return tuple(np.array(mc.to_rgb(c))*.79) if c in [C[k] for k in ["sc","fsm","tt","scalar"]] else c
def on_fill(c):
    rgb=np.array(mc.to_rgb(c))
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    luminance=float(np.dot(linear,[.2126,.7152,.0722]))
    return "white" if 1.05/(luminance+.05)>=4.5 else C["ink"]
def canvas(h):
    global TEXTS
    TEXTS=[]
    f=plt.figure(figsize=(W/72,h/72),facecolor="white")
    a=f.add_axes([0,0,1,1]);a.set(xlim=(0,W),ylim=(h,0),aspect="equal");a.set_axis_off()
    return f,a
def txt(a,x,y,s,size=7,color=None,ha="left",va="center",weight="normal",rotation=0,panel=False):
    size=10 if panel else (8.5 if weight=="bold" else max(7.5,min(8.5,size+1.0)))
    if weight=="normal" and "$" not in s and color!=C["muted"]:
        weight="semibold"
    t=a.text(x,y,s,fontsize=size,color=color or C["ink"],ha=ha,va=va,weight=weight,
             rotation=rotation,linespacing=1.3,zorder=10)
    TEXTS.append(t);return t
def line(a,pts,color=None,lw=.8,style="-",z=2):
    a.plot(*zip(*pts),color=edge(color or C["line"]),lw=lw*1.12,ls=style,zorder=z,
           solid_capstyle="round",solid_joinstyle="round")
def arrow(a,pts,color=None,lw=.85,head=5,style="-",rad=None):
    c=edge(color or C["line"])
    lw*=1.14
    if len(pts)>2:line(a,pts[:-1],c,lw,style)
    p,q=pts[-2:]
    kw={"connectionstyle":f"arc3,rad={rad}"} if rad is not None else {}
    a.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>",mutation_scale=head,lw=lw,color=c,
                shrinkA=0,shrinkB=0,linestyle=style,zorder=4,**kw))
def box(a,x,y,w,h,color=None,fill=None,lw=.75,r=0):
    semantic=color in [C[k] for k in ["sc","fsm","tt","scalar"]]
    if semantic and w>=70 and h>=20 and isinstance(fill,tuple):
        fill=tint(color,.85)
        lw=max(lw,.9)
    kw=dict(linewidth=lw,edgecolor=edge(color or C["line"]),facecolor=fill or "white")
    p=FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",**kw) if r else Rectangle((x,y),w,h,**kw)
    a.add_patch(p);return p
def dot(a,x,y,color=None,r=1.65):
    a.add_patch(Circle((x,y),r,fc=color or C["sc"],ec="none",zorder=5))
def header(a,x,y,w,letter,title):
    txt(a,x,y,letter,8,weight="bold",panel=True);txt(a,x+15,y,title,7,weight="bold")
    line(a,[(x,y+11),(x+w,y+11)],C["rule"],.65)
def fraction(a,x,y,numerator,denominator,size=7,pad=3.2):
    """Set a vector fraction with explicit clearance above/below its rule."""
    top=txt(a,x,y-pad,numerator,size,ha="center",va="bottom")
    bottom=txt(a,x,y+pad,denominator,size,ha="center",va="top")
    a.figure.canvas.draw()
    renderer=a.figure.canvas.get_renderer()
    width=max(top.get_window_extent(renderer).width,bottom.get_window_extent(renderer).width)*72/a.figure.dpi+4
    line(a,[(x-width/2,y),(x+width/2,y)],C["ink"],.45)
def tensor(a,x,y,w=37,h=30,color=None,rows=4,cols=4,stack=2,values=False):
    c=color or C["sc"]
    for k in range(stack,0,-1):box(a,x+2.2*k,y-2.2*k,w,h,c,tint(c,.91),.65)
    for i in range(rows):
        for j in range(cols):
            pale=.16+.55*((i*7+j*3+2)%11)/10 if values else .16
            box(a,x+j*w/cols,y+i*h/rows,w/cols,h/rows,"white",tint(c,pale),.35)
    box(a,x,y,w,h,c,"none",.75)
def tt_tile(a,x,y,w,h,rows,cols,stack=0):tensor(a,x,y,w,h,C["tt"],rows,cols,stack)
def bits(a,x,y,seq,cell=7,h=9,color=None,labels=False):
    c=color or C["sc"]
    for i,b in enumerate(seq):
        box(a,x+i*cell,y,cell-.8,h,c if b else C["rule"],c if b else "white",.5)
        if labels:txt(a,x+i*cell+(cell-.8)/2,y+h/2,str(b),6.2,on_fill(c) if b else C["muted"],ha="center")
def state(a,x,y,label,selected=False,r=8,color=None):
    c=color or C["fsm"]
    a.add_patch(Circle((x,y),r,fc=c if selected else tint(c,.86),ec=edge(c),lw=.9,zorder=5))
    txt(a,x,y,str(label),6.8,on_fill(c) if selected else C["ink"],ha="center")
def add(a,x,y,color=None,r=6.5):
    c=color or C["sc"];a.add_patch(Circle((x,y),r,fc="white",ec=c,lw=.85,zorder=5))
    line(a,[(x-r*.55,y),(x+r*.55,y)],c,.8,z=6);line(a,[(x,y-r*.55),(x,y+r*.55)],c,.8,z=6)
def minitt(a,x,y,w=43):
    for dx in [0,w/2,w]:box(a,x+dx,y,7,7,C["tt"],tint(C["tt"],.84),.65)
    line(a,[(x+7,y+3.5),(x+w/2,y+3.5)],C["tt"],.8);line(a,[(x+w/2+7,y+3.5),(x+w,y+3.5)],C["tt"],.8)
def compact(f,name):
    heights={"overview":285,"representation":365,"operator":510,"block":520}
    oldh=f.get_figheight()*72
    newh=heights[name]
    scale=newh/oldh
    a=f.axes[0]
    # Reposition vector geometry while keeping type and circular nodes undistorted.
    for t in a.texts:
        x,y=t.get_position();t.set_position((x,y*scale))
    for ln in a.lines:
        ln.set_ydata(np.asarray(ln.get_ydata(),dtype=float)*scale)
    for p in a.patches:
        if isinstance(p,Circle):
            x,y=p.center;p.center=(x,y*scale)
        elif isinstance(p,Rectangle):
            p.set_y(p.get_y()*scale);p.set_height(p.get_height()*scale)
        elif isinstance(p,FancyBboxPatch):
            p.set_y(p.get_y()*scale);p.set_height(p.get_height()*scale)
        elif isinstance(p,FancyArrowPatch):
            if p._posA_posB is not None:
                p._posA_posB=[(x,y*scale) for x,y in p._posA_posB]
            else:
                p._path_original.vertices[:,1]*=scale
        elif isinstance(p,Polygon):
            v=p.get_xy();v[:,1]*=scale;p.set_xy(v)
        elif isinstance(p,PathPatch):
            p.get_path().vertices[:,1]*=scale
        else:raise TypeError(type(p))
    f.set_size_inches(W/72,newh/72)
    a.set_ylim(newh,0)

def finish(f,name):
    compact(f,name)
    f.canvas.draw();renderer=f.canvas.get_renderer();lim=f.bbox;outside=[]
    for t in TEXTS:
        b=t.get_window_extent(renderer)
        if b.x0<lim.x0-1 or b.x1>lim.x1+1 or b.y0<lim.y0-1 or b.y1>lim.y1+1:outside.append(t.get_text())
    if outside:raise ValueError((name,"text outside page",outside))
    p=OUT/f"Fig_FROST_{name}.pdf"
    staged=TMP/f"export_{name}.pdf"
    f.savefig(staged,metadata={"Title":f"FROST: {name}","Author":"FROST authors",
                         "Subject":"Vector scientific diagram; embedded editable text"})
    staged.replace(p)
    f.savefig(TMP/f"{name}_compact_preview.png",dpi=160);plt.close(f)
    d=fitz.open(p);assert not d[0].get_images();assert len(d[0].get_text())>200
    print(name,len(d[0].get_drawings()),len(d[0].get_text()),"vector paths / text characters")

def overview():
    f,a=canvas(310)
    header(a,12,15,181,"a","Coupled nonlinear mappings")
    header(a,207,15,291,"b","Tensorized finite-state representation")
    txt(a,102,42,"Input scores or features",7,ha="center")
    xs=[31,71,111,174]
    for xx in xs:
        for yy in xs:line(a,[(xx,69),(yy,116)],C["rule"],.5)
    for i,x in enumerate(xs):
        sub=["1","2","3","M"][i]
        for y,letter,c in [(62,"x",C["sc"]),(123,"y",C["fsm"])]:
            a.add_patch(Circle((x,y),11,fc=tint(c,.90),ec=c,lw=.8,zorder=5))
            txt(a,x,y,"$"+letter+"_{"+sub+"}$",8,ha="center")
    txt(a,142,62,r"$\cdots$",9,ha="center");txt(a,142,123,r"$\cdots$",9,ha="center")
    txt(a,102,151,"Softmax / feature normalization",6.8,ha="center")
    for x,label in [(231,"Bitstreams"),(302,"FSM bank"),(424,"State-selected TT slices")]:txt(a,x,42,label,6.8,ha="center")
    for y,seq in [(56,[1,1,0,1,0,0,1]),(74,[1,0,0,1,1,0,0]),(109,[1,1,0,1,1,0,1])]:bits(a,209,y,seq,6,7)
    txt(a,230,96,r"$\vdots$",9,ha="center");arrow(a,[(256,82),(268,82)])
    for j,y in enumerate([60,82,113]):
        line(a,[(277,y),(325,y)],C["rule"],.65)
        for k,x in enumerate([277,293,309,325]):
            a.add_patch(Circle((x,y),3.6,fc=C["fsm"] if k==[1,1,3][j] else "white",
                ec=C["fsm"] if k==[1,1,3][j] else C["line"],lw=.6,zorder=5))
    txt(a,299,98,r"$\vdots$",8,ha="center")
    line(a,[(334,55),(338,55),(338,118),(334,118)],C["line"],.6)
    txt(a,301,130,r"Joint state $\mathbf{s}=(i_1,\ldots,i_M)$",6.2,ha="center")
    arrow(a,[(342,82),(353,82)])
    tt_tile(a,357,79,19,7,1,3);tt_tile(a,390,72,20,21,3,3);tt_tile(a,447,72,7,21,3,1)
    arrow(a,[(378,82),(388,82)],C["tt"],.7,4);line(a,[(412,82),(422,82)],C["tt"],.7)
    txt(a,432,82,r"$\cdots$",9,C["tt"],ha="center");arrow(a,[(458,82),(473,82)],C["tt"],.7,4)
    box(a,478,77,10,10,C["tt"],C["tt"],.6);txt(a,483,62,r"$\bar{w}_{\mathbf{s}}$",8,ha="center")
    for x,lab,dim in [(367,r"$G^{(1)}[i_1]$",r"$1\times r_1$"),(400,r"$G^{(2)}[i_2]$",r"$r_1\times r_2$"),(451,r"$G^{(M)}[i_M]$",r"$r_{M-1}\times1$")]:
        txt(a,x,106,lab,5.9,ha="center");txt(a,x,118,dim,5.5,C["muted"],ha="center")
    txt(a,423,137,"One output component",6.3,ha="center")
    txt(a,273,154,r"$P_y=\sum_{\mathbf{s}}P_{\mathbf{s}}\,\bar{w}_{\mathbf{s}}$",8,ha="center")
    txt(a,424,154,r"$N^M\longrightarrow O(MNr^2)$",8,ha="center")
    txt(a,424,167,"Storage per output; bounded TT rank",5.9,C["muted"],ha="center")
    header(a,12,180,486,"c","From stochastic operators to a Transformer")
    tensor(a,15,234,30,31);txt(a,31,216,r"Input $X$",7,ha="center")
    tensor(a,465,234,29,31);txt(a,480,216,r"Output $Y$",7,ha="center")
    box(a,85,218,137,61,C["rule"],tint(C["sc"],.955),.7);box(a,284,218,137,61,C["rule"],tint(C["sc"],.955),.7)
    txt(a,154,230,"Attention sublayer",7.2,ha="center",weight="bold")
    txt(a,353,230,"Feed-forward sublayer",7,ha="center",weight="bold")
    for x in [96,295]:
        line(a,[(x,241),(x,254)],C["sc"],2.2);line(a,[(x,259),(x,272)],C["fsm"],2.2)
    txt(a,103,247,"SC projections and\nattention products",6.2);txt(a,103,265,"Finite-state softmax\nand normalization",6.2)
    txt(a,302,247,"SC linear\ntransformations",6.2);txt(a,302,265,"Finite-state activation\nand normalization",6.2)
    for pts in [[(49,250),(85,250)],[(222,250),(240,250)],[(253,250),(284,250)],[(421,250),(439,250)],[(452,250),(465,250)]]:arrow(a,pts)
    add(a,247,250,C["line"]);add(a,446,250,C["line"])
    arrow(a,[(65,250),(65,208),(247,208),(247,243)],C["line"],.65,4)
    arrow(a,[(267,250),(267,208),(446,208),(446,243)],C["line"],.65,4)
    for x in [154,353]:
        txt(a,x,200,"Residual",6.3,C["muted"],ha="center")
    txt(a,253,289,"Compose across layers; retain scaling and representation interfaces",6.3,C["muted"],ha="center")
    for x,s,c,lab in [(72,"L",C["sc"],"Bitstream length"),(216,"N",C["fsm"],"States per FSM"),(362,"r",C["tt"],"Maximum TT rank")]:
        txt(a,x,303,"$"+s+"$",7.5,c,weight="bold");txt(a,x+12,303,lab,6.3)
    finish(f,"overview")

def gate_and(a,x,y,c):
    path=MPath([(x,y-15),(x+14,y-15),(x+35,y-15),(x+35,y+15),(x+14,y+15),(x,y+15),(x,y-15)],
               [MPath.MOVETO,MPath.LINETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO,MPath.CLOSEPOLY])
    a.add_patch(PathPatch(path,fc=tint(c,.89),ec=c,lw=.9))
def gate_xnor(a,x,y,c):
    v=[(x,y-15),(x+14,y-15),(x+25,y-10),(x+34,y),(x+25,y+10),(x+14,y+15),(x,y+15),(x+8,y+5),(x+8,y-5),(x,y-15)]
    a.add_patch(PathPatch(MPath(v,[MPath.MOVETO]+[MPath.CURVE4]*9),fc=tint(c,.89),ec=c,lw=.9))
    a.add_patch(PathPatch(MPath([(x-4,y-15),(x+4,y-5),(x+4,y+5),(x-4,y+15)],[MPath.MOVETO]+[MPath.CURVE4]*3),fill=False,ec=c,lw=.9))
    a.add_patch(Circle((x+37,y),2.5,fc="white",ec=c,lw=.9))

def representation():
    f,a=canvas(420)
    header(a,12,15,486,"a","From numerical tensors to stochastic streams")
    for x,n,lab in [(22,"1","Numerical tensor"),(184,"2","Range encoding"),(366,"3","Serial bitstreams")]:
        txt(a,x,43,n,7.3,C["muted"],weight="bold");txt(a,x+12,43,lab,7.2,weight="bold")
    tensor(a,33,70,46,35,stack=2,values=True);txt(a,57,122,r"$X\in\mathbb{R}^{n\times d}$",7.8,ha="center")
    arrow(a,[(94,89),(147,89)])
    txt(a,211,76,r"$p=$",7)
    fraction(a,244,75,r"$x-x_{\min}$",r"$x_{\max}-x_{\min}$",7)
    line(a,[(184,107),(295,107)],C["rule"],2);line(a,[(184,107),(240,107)],C["sc"],2);dot(a,240,107,C["sc"],3)
    txt(a,184,120,"0",6.2,C["muted"],ha="center");txt(a,295,120,"1",6.2,C["muted"],ha="center")
    arrow(a,[(313,89),(338,89)])
    for y,c,seq in [(66,C["sc"],[1,0,1,1,0,0,1,0]),(87,C["fsm"],[0,0,1,0,0,0,1,0]),(108,C["tt"],[0,1,0,0,0,1,0,0])]:
        xx=[362];yy=[y+8]
        for i,b in enumerate(seq):xx.extend([362+i*15,362+(i+1)*15]);yy.extend([y+8*(1-b),y+8*(1-b)])
        line(a,list(zip(xx,yy)),c,.95);line(a,[(362,y+10),(482,y+10)],C["rule"],.5)
    arrow(a,[(362,128),(484,128)],C["line"],.65,4);txt(a,491,128,r"$t$",7,C["muted"])
    line(a,[(12,143),(498,143)],C["rule"],.6)
    for x,title,c,pval,domain,seq,eq in [(22,"Unipolar",C["sc"],r"$p=x=0.5$",r"$x\in[0,1]$",[1,0,1,1,0,0,1,0],r"$\hat{x}=4/8=0.5$"),(280,"Bipolar",C["fsm"],r"$p=(x+1)/2=0.75$",r"$x\in[-1,1]$",[1,1,0,1,1,1,0,1],r"$\hat{x}=2(6/8)-1=0.5$")]:
        txt(a,x,158,title,7.5,weight="bold");txt(a,x+121,158,pval,7,ha="center");txt(a,x,171,domain,6.4,C["muted"])
        bits(a,x,181,seq,10.4,13,c,True);txt(a,x+94,188,eq,6.8)
    txt(a,497,208,"Illustrative 8-bit windows; normalized value x = 0.5",6.2,C["muted"],ha="right")
    header(a,12,229,486,"b","Gate-level stochastic arithmetic")
    for x,title in [(90,"Unipolar product"),(255,"Bipolar product"),(421,"Weighted average")]:txt(a,x,254,title,7.3,ha="center",weight="bold")
    line(a,[(172,251),(172,326)],C["rule"],.6);line(a,[(337,251),(337,326)],C["rule"],.6)
    gate_and(a,75,287,C["sc"])
    for y,lab in [(280,r"$x_b$"),(294,r"$y_b$")]:txt(a,28,y,lab,7);arrow(a,[(44,y),(75,y)],C["sc"],.85,4)
    arrow(a,[(107,287),(139,287)],C["sc"],.85,4);txt(a,147,287,r"$z_b$",7)
    txt(a,90,313,"AND",7.1,ha="center");txt(a,90,328,r"$z=xy$",7.5,ha="center")
    gate_xnor(a,240,287,C["fsm"])
    for y,lab in [(280,r"$x_b$"),(294,r"$y_b$")]:txt(a,193,y,lab,7);arrow(a,[(209,y),(242,y)],C["fsm"],.85,4)
    arrow(a,[(280,287),(308,287)],C["fsm"],.85,4);txt(a,316,287,r"$z_b$",7)
    txt(a,255,313,"XNOR",7.1,ha="center");txt(a,255,328,r"$z=xy$",7.5,ha="center")
    a.add_patch(Polygon([(408,273),(435,279),(435,304),(408,310)],closed=True,fc=tint(C["tt"],.89),ec=C["tt"],lw=.9))
    for y,lab,bit in [(282,r"$x_b$","1"),(300,r"$y_b$","0")]:
        txt(a,358,y,lab,7);arrow(a,[(375,y),(408,y)],C["tt"],.85,4);txt(a,413,y,bit,5.5,C["muted"])
    arrow(a,[(449,264),(420,264),(420,276)],C["tt"],.8,4);txt(a,455,264,r"$c_b$",7)
    arrow(a,[(435,291),(473,291)],C["tt"],.85,4);txt(a,480,291,r"$z_b$",7)
    txt(a,421,319,"MUX",7.1,ha="center");txt(a,421,333,r"$z=\alpha x+(1-\alpha)y$",7,ha="center")
    txt(a,12,346,r"Products: independent streams. MUX: selector independent of its inputs; $\Pr(c_b=1)=\alpha$.",6,C["muted"])
    header(a,12,367,486,"c","Retain the numerical scale with each stream")
    for x,w,c in [(12,144,C["sc"]),(191,125,C["tt"]),(351,147,C["sc"])]:box(a,x,387,w,29,c,tint(c,.95),.75)
    txt(a,84,396,r"MUX output ($\alpha=1/2$)",6.4,ha="center");txt(a,84,408,r"$m=(x+y)/2$",7.1,ha="center")
    txt(a,253,396,"Scale metadata",6.6,ha="center");txt(a,253,408,r"$g=2$",7.3,ha="center")
    txt(a,424,396,"Represented sum",6.6,ha="center");txt(a,424,408,r"$s=gm=x+y$",7.1,ha="center")
    arrow(a,[(158,401),(189,401)]);arrow(a,[(318,401),(349,401)])
    finish(f,"representation")

def operator():
    f,a=canvas(600)
    header(a,12,15,486,"a","Input-dependent finite-state basis")
    txt(a,127,45,r"Bit 1: step right ($x_m$)",7,ha="center")
    pos=[37,97,157,217]
    for i in range(3):
        arrow(a,[(pos[i]+9,76),(pos[i+1]-9,76)],C["fsm"],.9,4.5,rad=-.52)
        arrow(a,[(pos[i+1]-9,86),(pos[i]+9,86)],C["sc"],.9,4.5,rad=-.52)
    for x,i in zip(pos,range(4)):state(a,x,81,i,i==2,r=10)
    for vertices,c in [([(29,74),(5,60),(5,102),(29,88)],C["sc"]),
                       ([(225,88),(250,102),(250,60),(225,74)],C["fsm"])]:
        a.add_patch(FancyArrowPatch(path=MPath(vertices,[MPath.MOVETO]+[MPath.CURVE4]*3),
                    arrowstyle="-|>",mutation_scale=4.5,color=c,lw=.9,zorder=4))
    txt(a,127,116,r"Bit 0: step left ($1-x_m$)",7,ha="center")
    txt(a,19,134,"Saturating random walk",6.2,C["muted"])
    txt(a,128,146,r"$x_m=\Pr[x_{b,m}(t)=1]$",7.2,ha="center")
    txt(a,60,175,r"$\pi_m(i\mid x_m)=$",7)
    fraction(a,112,174,r"$r_m^i$",r"$\sum_{u=0}^{N-1}r_m^u$",7,pad=3.5)
    txt(a,132,175,r"$,\quad r_m=$",7)
    fraction(a,176,174,r"$x_m$",r"$1-x_m$",7,pad=3.5)
    txt(a,127,211,r"$0<x_m<1$; endpoint values are defined by limits",6.1,C["muted"],ha="center")
    line(a,[(264,39),(264,202)],C["rule"],.65)
    # Exact stationary occupancy of the four-state reflecting random walk.
    for j,xm in enumerate([.25,.5,.75]):
        x=288+j*72
        txt(a,x+26,45,"$x_m="+str(xm)+"$",7.1,ha="center")
        line(a,[(x,61),(x+52,61)],C["rule"],.55)
        line(a,[(x,113),(x+52,113)],C["line"],.6)
        rm=xm/(1-xm);pr=rm**np.arange(4);pr=pr/pr.sum()
        for i,v in enumerate(pr):
            box(a,x+3+i*12,113-v*52,8,v*52,"none",tint(C["fsm"],.68-.19*i),0)
            txt(a,x+7+i*12,121,str(i),5.8,C["muted"],ha="center")
    txt(a,282,61,"1",5.9,C["muted"],ha="right");txt(a,282,113,"0",5.9,C["muted"],ha="right")
    txt(a,273,87,"Probability",6.1,C["muted"],rotation=90,ha="center")
    txt(a,395,138,r"Analytic stationary occupancies; $N=4$",6.1,C["muted"],ha="center")
    txt(a,395,155,"Joint stationary occupancy",6.8,ha="center")
    txt(a,395,178,r"$P_{\mathbf{s}}=\prod_{m=1}^{M}\pi_m(i_m\mid x_m)$",8.4,ha="center")
    txt(a,395,201,"Temporal IID bits; mutually independent streams",5.9,C["muted"],ha="center")
    header(a,12,227,486,"b","State selection and tensor-train contraction")
    for x,sub,chosen in [(28,"1",1),(208,"2",2),(404,"M",3)]:
        txt(a,x,255,"$x_{b,"+sub+"}(t)$",6.9)
        bits(a,x+58,248,[1,0,1,1,0,1],6.8,8,C["fsm"])
        arrow(a,[(x+78,258),(x+78,268)],C["fsm"],.8,4)
        box(a,x,272,91,22,C["rule"],"white",.65)
        line(a,[(x+15,283),(x+78,283)],C["fsm"],.75)
        for i in range(4):state(a,x+15+21*i,283,i,chosen==i,4.6)
        tx,ty,labx,laby={"1":(59,321,75,315),"2":(236,312,269,315),"M":(426,313,448,316)}[sub]
        sx=x+15+21*chosen
        arrow(a,[(sx,289),(sx,305),(tx,305),(tx,ty)],C["fsm"],.8,4)
        txt(a,labx,laby,"$i_"+sub+"(t)$",6.3)
    txt(a,366,282,r"$\cdots$",10,C["muted"],ha="center")
    tt_tile(a,35,330,48,11,1,4,3);tt_tile(a,217,321,38,37,4,4,3)
    tt_tile(a,421,322,11,36,4,1,3)
    line(a,[(83,336),(214,336)],C["tt"],1)
    line(a,[(258,336),(307,336)],C["tt"],1)
    txt(a,338,336,r"$\cdots$",10,C["tt"],ha="center")
    line(a,[(365,336),(419,336)],C["tt"],1)
    arrow(a,[(436,336),(475,336)],C["tt"],1,5)
    box(a,482,331,10,10,C["tt"],C["tt"],.6)
    txt(a,487,314,r"$\bar{w}_{\mathbf{s}}$",7.5,ha="center")
    txt(a,486,354,"One output\ncomponent",5.7,C["muted"],ha="center")
    for x,sub,dim in [(60,"1",r"$1\times r_1$"),(236,"2",r"$r_1\times r_2$"),(426,"M",r"$r_{M-1}\times1$")]:
        txt(a,x,370,"$G^{("+sub+")}[i_"+sub+"(t)]$",6.6,ha="center")
        txt(a,x,383,dim,6,C["muted"],ha="center")
    for x,lab in [(147,r"$r_1$"),(284,r"$r_2$"),(393,r"$r_{M-1}$")]:txt(a,x,327,lab,6.3,ha="center")
    txt(a,19,400,r"State-selected slices: $\mathbf{s}(t)=(i_1(t),\ldots,i_M(t))$",6.2,C["muted"])
    txt(a,374,400,r"Stationary mean: $P_y=\sum_{\mathbf{s}}P_{\mathbf{s}}\,\bar{w}_{\mathbf{s}}$",7,ha="center")
    header(a,12,423,271,"c","Admissible stochastic emission")
    header(a,306,423,192,"d","Scalar parameter storage")
    txt(a,16,447,r"Scalar: $0\leq\bar{w}_{\mathbf{s}}\leq1$  (Bernoulli)",6.7)
    txt(a,16,464,r"Attention: $\bar{w}_{\mathbf{s},c}\geq0,\quad\sum_{c=1}^{C}\bar{w}_{\mathbf{s},c}=1$",6.7)
    txt(a,21,491,"One event per cycle",6.8,weight="bold")
    txt(a,189,491,r"Frequency ($L=8$)",6.8,weight="bold")
    events=[0,2,1,2,3,0,3,2]
    for i,c in enumerate(CAT):
        txt(a,16,510+10*i,str(i+1),5.7,ha="center")
        for j,evt in enumerate(events):box(a,24+10*j,506+10*i,8,8,c if evt==i else C["rule"],c if evt==i else "white",.4)
        count=events.count(i)
        box(a,190,506+10*i,count*11,8,"none",c,0);txt(a,193+count*11,510+10*i,str(count)+"/8",5.9)
    arrow(a,[(112,525),(165,525)],C["line"],.75,4)
    txt(a,64,558,r"$\sum_{c=1}^{C} I_c(t)=1$",7,ha="center")
    txt(a,222,558,r"$\sum_{c=1}^{C}\hat{a}_c=1$",7,ha="center")
    txt(a,144,577,r"$\hat{a}_c=(1/L)\sum_{t=1}^{L}I_c(t)$",6.8,ha="center")
    # Analytic storage counts, exactly matching the original figure.
    ax=f.add_axes([338/W,63/600,150/W,78/600])
    M=np.array([2,4,8,16]);N=16;r=8
    ax.plot(M,N**M.astype(float),color="#808080",lw=1.1,label=r"Dense $N^M$")
    ax.plot(M,2*N*r+(M-2)*N*r*r,color=C["tt"],lw=1.2,marker="o",ms=2.4,label="TT")
    ax.set(yscale="log",xlim=(2,16),ylim=(1e2,1e20),xticks=[2,8,16],yticks=[1e2,1e8,1e14,1e20])
    ax.tick_params(labelsize=7.5,pad=2,length=2.8,width=.75,colors=C["muted"])
    ax.spines[["top","right"]].set_visible(False)
    for sp in ["left","bottom"]:ax.spines[sp].set_color(C["line"])
    ax.set_xlabel(r"Inputs, $M$",fontsize=7.5,labelpad=2,color=C["ink"])
    ax.set_ylabel("Parameters per output",fontsize=7.5,labelpad=3,color=C["ink"])
    ax.legend(frameon=False,fontsize=7.5,loc="upper left",handlelength=1.8,labelspacing=.4,borderpad=.2)
    txt(a,488,447,r"$N=16,\ r=8$",6.2,C["muted"],ha="right")
    txt(a,406,566,r"TT: $2Nr+(M-2)Nr^2$",6.8,ha="center")
    txt(a,406,579,r"$r_0=r_M=1$; all internal ranks $r$",5.8,C["muted"],ha="center")
    finish(f,"operator")

def block():
    f,a=canvas(645)
    header(a,12,15,135,"a","Complete block")
    header(a,168,15,330,"b","Inside multihead attention")
    # Complete post-normalization architecture.
    tensor(a,66,48,34,29,values=True);txt(a,45,63,r"$X$",8)
    txt(a,83,37,r"$n\times d$",6.2,C["muted"],ha="center")
    arrow(a,[(83,78),(83,102)],C["sc"]);dot(a,83,90)
    box(a,32,102,102,61,C["sc"],tint(C["sc"],.95),.8)
    txt(a,83,114,"Multihead",7.3,ha="center",weight="bold")
    txt(a,83,129,"SC attention",7,ha="center")
    for x in [49,77,105]:tensor(a,x,143,13,12,stack=1,rows=2,cols=2)
    arrow(a,[(83,163),(83,184)],C["sc"]);txt(a,95,176,r"$O$",7)
    add(a,83,191)
    arrow(a,[(83,90),(18,90),(18,191),(76,191)],C["sc"],.8)
    txt(a,10,140,"Residual",6.1,C["muted"],rotation=90,ha="center")
    arrow(a,[(83,198),(83,218)],C["sc"])
    box(a,32,218,102,48,C["fsm"],tint(C["fsm"],.93),.8)
    txt(a,83,232,"Tensorized norm 1",7.0,ha="center",weight="bold");minitt(a,63,249,33)
    arrow(a,[(83,266),(83,314)],C["sc"]);dot(a,83,287)
    txt(a,97,287,r"$H$",7.5)
    box(a,32,314,102,61,C["sc"],tint(C["sc"],.95),.8)
    txt(a,83,328,"SC feed-forward",7,ha="center",weight="bold")
    for x,h,c in [(54,11,C["sc"]),(81,22,C["scalar"]),(108,11,C["sc"])]:
        box(a,x,354-h/2,7,h,"none",tint(c,.27),0)
    arrow(a,[(64,354),(78,354)],C["sc"],.7,4);arrow(a,[(91,354),(105,354)],C["sc"],.7,4)
    arrow(a,[(83,375),(83,405)],C["sc"]);txt(a,96,391,r"$F$",7.5)
    add(a,83,412)
    arrow(a,[(83,287),(18,287),(18,412),(76,412)],C["sc"],.8)
    txt(a,10,351,"Residual",6.1,C["muted"],rotation=90,ha="center")
    arrow(a,[(83,419),(83,441)],C["sc"])
    box(a,32,441,102,48,C["fsm"],tint(C["fsm"],.93),.8)
    txt(a,83,455,"Tensorized norm 2",7,ha="center",weight="bold");minitt(a,63,472,33)
    arrow(a,[(83,489),(83,527)],C["sc"])
    tensor(a,66,532,34,29,values=True);txt(a,45,547,r"$Y$",8)
    txt(a,112,548,r"$n\times d$",6.2,C["muted"])
    txt(a,83,580,r"$H=\mathrm{Norm}_1(X+O)$",6.7,ha="center")
    txt(a,83,594,r"$Y=\mathrm{Norm}_2(H+F)$",6.7,ha="center")
    # Links to the expanded sublayers are guides, not numerical data paths.
    arrow(a,[(135,133),(153,133),(153,40),(164,40)],C["line"],.6,4,style=(0,(3,3)))
    arrow(a,[(135,344),(153,344),(153,490),(164,490)],C["line"],.6,4,style=(0,(3,3)))
    txt(a,491,37,r"Per head $h$",6.4,C["muted"],ha="right")
    txt(a,333,41,r"$X$",8,ha="center")
    line(a,[(220,53),(446,53)],C["sc"],.8);line(a,[(333,47),(333,53)],C["sc"],.8)
    for cx,sub in [(220,"Q"),(333,"K"),(446,"V")]:
        arrow(a,[(cx,53),(cx,66)],C["sc"],.8)
        box(a,cx-38,66,76,38,C["sc"],tint(C["sc"],.95),.8)
        txt(a,cx,77,"SC linear",7.2,ha="center",weight="bold")
        txt(a,cx,93,"$W_{"+sub+"}^{(h)}$",8,ha="center")
        arrow(a,[(cx,104),(cx,117)],C["sc"],.8)
        tensor(a,cx-18,122,33,28,values=True)
        txt(a,cx+24,133,"$"+sub+"^{(h)}$",6.9)
        txt(a,cx+24,146,r"$n\times d_k$" if sub!="V" else r"$n\times d_v$",5.5,C["muted"])
    arrow(a,[(220,151),(220,168),(268,168),(268,181)],C["sc"],.85)
    arrow(a,[(333,151),(333,181)],C["sc"],.85)
    box(a,182,182,225,43,C["sc"],tint(C["sc"],.95),.8)
    txt(a,294,194,"SC query-key products",7.1,ha="center",weight="bold")
    txt(a,294,213,r"$Z^{(h)}=Q^{(h)}(K^{(h)})^{\mathsf{T}}/\sqrt{d_k}$",7,ha="center")
    arrow(a,[(220,225),(220,246)],C["sc"],.85)
    tensor(a,202,251,33,32,values=True)
    box(a,202,259,33,8,C["fsm"],"none",1.15)
    txt(a,219,298,r"Scores $Z^{(h)}$",6.5,ha="center")
    txt(a,219,311,r"$n\times n$",6,C["muted"],ha="center")
    txt(a,241,254,r"$z_i^{(h)}$",6.5)
    arrow(a,[(236,263),(273,263)],C["fsm"],.85)
    box(a,275,244,78,48,C["fsm"],tint(C["fsm"],.93),.8)
    txt(a,314,264,r"$\mathcal{T}_{\mathrm{att}}$",9,ha="center");minitt(a,292,278,36)
    txt(a,314,305,"Joint row mapping",6.7,ha="center")
    txt(a,314,323,r"$a_i^{(h)}=\mathcal{T}_{\mathrm{att}}(z_i^{(h)})$",7.2,ha="center")
    txt(a,314,348,r"$a_{ij}^{(h)}\geq0,\quad\sum_{j=1}^{n}a_{ij}^{(h)}=1$",7.2,ha="center")
    arrow(a,[(353,263),(420,263)],C["fsm"],.85)
    txt(a,384,254,r"$a_i^{(h)}$",6.5,ha="center")
    tensor(a,423,251,33,32,C["fsm"],values=True)
    txt(a,439,237,r"Weights $A^{(h)}$",6.5,ha="center")
    txt(a,451,298,r"$n\times n$",6,C["muted"])
    # Values bypass softmax; attention weights and V enter aggregation.
    arrow(a,[(446,151),(493,151),(493,396),(482,396)],C["sc"],.8,4.5)
    txt(a,482,205,r"$V^{(h)}$",6.5,C["muted"],ha="right")
    arrow(a,[(439,284),(439,367)],C["fsm"],.85,4.5)
    box(a,278,368,204,65,C["sc"],tint(C["sc"],.95),.8)
    txt(a,380,378,"SC value aggregation",7.2,ha="center",weight="bold")
    txt(a,380,393,r"$O^{(h)}=A^{(h)}V^{(h)}$",7.6,ha="center")
    txt(a,380,417,r"$o_i^{(h)}=\sum_{j=1}^{n}a_{ij}^{(h)}v_j^{(h)}$",7.4,ha="center")
    tensor(a,202,381,33,30,values=True)
    arrow(a,[(278,396),(240,396)],C["sc"],.8,4.5)
    txt(a,217,371,r"Head outputs $O^{(h)}$",6.4,ha="center")
    txt(a,196,400,r"$n\times d_v$",5.8,C["muted"],ha="right")
    arrow(a,[(219,412),(219,436),(333,436),(333,445)],C["sc"],.8,4.5)
    box(a,180,447,309,33,C["sc"],tint(C["sc"],.95),.8)
    txt(a,334,456,r"Concatenate heads + SC output projection $W_O$",6.4,ha="center")
    txt(a,334,469,r"$O=\mathrm{Concat}_h(O^{(h)})W_O,\quad O\in\mathbb{R}^{n\times d}$",6.8,ha="center")
    header(a,168,498,330,"c","Inside the feed-forward transformation")
    for x,title,c in [(180,"SC linear",C["sc"]),(296,"Scalar FSM",C["scalar"]),(412,"SC linear",C["sc"])]:
        box(a,x,523,77,53,c,tint(c,.95),.8)
        txt(a,x+38,535,title,7,ha="center",weight="bold")
    tensor(a,192,550,18,15,rows=2,cols=2,stack=1);txt(a,233,558,r"$W_1$",7.5,ha="center")
    tensor(a,424,550,18,15,rows=2,cols=2,stack=1);txt(a,465,558,r"$W_2$",7.5,ha="center")
    # GELU target curve; schematic axes omitted as in the original panel.
    xx=np.linspace(-2,2,80);gelu=.5*xx*(1+np.vectorize(math.erf)(xx/np.sqrt(2)))
    line(a,list(zip(306+(xx+2)*13,566-gelu*10)),C["scalar"],1.1)
    arrow(a,[(257,553),(296,553)],C["sc"],.85);arrow(a,[(373,553),(412,553)],C["sc"],.85)
    txt(a,218,587,r"$W_1:d\times d_{\mathrm{ff}}$",6.5,C["muted"],ha="center")
    txt(a,334,587,r"$\mathcal{T}_{\mathrm{act}}$: elementwise GELU",6.2,C["muted"],ha="center")
    txt(a,450,587,r"$W_2:d_{\mathrm{ff}}\times d$",6.5,C["muted"],ha="center")
    txt(a,334,603,r"$F=\mathrm{SCLinear}_2(\mathcal{T}_{\mathrm{act}}(\mathrm{SCLinear}_1(H)))$",7,ha="center")
    txt(a,255,621,"Post-normalization; tensor tiles and colours are schematic.",6.2,C["muted"],ha="center")
    line(a,[(12,630),(498,630)],C["rule"],.6)
    for x,key,lab in [(15,"sc","SC arithmetic"),(139,"fsm","Tensorized FSM"),(273,"tt","TT parameters"),(394,"scalar","Scalar FSM")]:
        box(a,x,637,6,5,"none",C[key],0);txt(a,x+10,639.5,lab,6.2)
    finish(f,"block")

if __name__=="__main__":
    for name in (sys.argv[1:] or ["overview","representation","operator","block"]):
        globals()[name]()
