import os
import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="MindVault · Mental Health Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── LOAD MODEL ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("mental_health_model.pkl")
    return model

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@200;300;400;500;600;700;800;900&family=Space+Mono:wght@400;700&display=swap');

*, *::before, *::after { margin:0; padding:0; box-sizing:border-box; }

html, body, .stApp {
    background: #03050f !important;
    font-family: 'Sora', sans-serif;
    overflow-x: hidden;
}

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}
header { visibility: hidden !important; }

.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── MAIN SHELL ─────────────────────────────────────────────────────────── */
.shell {
    position: relative;
    z-index: 1;
    max-width: 960px;
    margin: 0 auto;
    padding: 72px 40px 120px;
}

/* ── NAV ────────────────────────────────────────────────────────────────── */
.nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 80px;
    padding-bottom: 24px;
    border-bottom: 1px solid rgba(99,179,237,0.08);
    animation: fadeDown 0.9s cubic-bezier(.16,1,.3,1) both;
}
@keyframes fadeDown {
    from { opacity:0; transform:translateY(-16px); }
    to   { opacity:1; transform:translateY(0); }
}
.nav-logo {
    font-family: 'Sora', sans-serif;
    font-weight: 800;
    font-size: 20px;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #63b3ed 0%, #4fd1c5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.nav-pill {
    display: flex;
    align-items: center;
    gap: 7px;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #4fd1c5;
    background: rgba(79,209,197,0.08);
    border: 1px solid rgba(79,209,197,0.18);
    padding: 8px 16px;
    border-radius: 40px;
}
.nav-pill-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #4fd1c5;
    box-shadow: 0 0 10px #4fd1c5;
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:.3; transform:scale(.7); }
}

/* ── HERO ───────────────────────────────────────────────────────────────── */
.hero {
    text-align: center;
    margin-bottom: 80px;
    animation: fadeUp 1s cubic-bezier(.16,1,.3,1) .1s both;
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(28px); }
    to   { opacity:1; transform:translateY(0); }
}
.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 9px 20px;
    background: rgba(99,179,237,0.08);
    border: 1px solid rgba(99,179,237,0.18);
    border-radius: 40px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 36px;
}
.hero-tag-glow {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #63b3ed;
    box-shadow: 0 0 12px #63b3ed;
    animation: blink 2.4s ease-in-out infinite;
}
.hero-h1 {
    font-size: clamp(2.6rem, 7vw, 4.8rem);
    font-weight: 900;
    line-height: 1.07;
    color: #f0f4ff;
    letter-spacing: -2.5px;
    margin-bottom: 24px;
}
.hero-h1 span {
    background: linear-gradient(135deg, #63b3ed 0%, #4fd1c5 45%, #68d391 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    background-size: 200% 200%;
    animation: shimmer 6s ease infinite;
}
@keyframes shimmer {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero-sub {
    font-size: 17px;
    color: #718096;
    line-height: 1.8;
    max-width: 520px;
    margin: 0 auto;
    font-weight: 300;
    animation: fadeUp 1s cubic-bezier(.16,1,.3,1) .2s both;
}
.hero-stats {
    display: flex;
    justify-content: center;
    gap: 56px;
    margin-top: 52px;
    flex-wrap: wrap;
    animation: fadeUp 1s cubic-bezier(.16,1,.3,1) .3s both;
}
.stat { display:flex; flex-direction:column; align-items:center; gap:6px; }
.stat-val {
    font-family: 'Space Mono', monospace;
    font-size: 26px;
    font-weight: 700;
    background: linear-gradient(135deg, #63b3ed, #4fd1c5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-lbl {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #4a5568;
    font-weight: 700;
}

/* ── DIVIDER ────────────────────────────────────────────────────────────── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,179,237,0.2), transparent);
    margin: 72px 0;
}

/* ── SECTION HEADER ─────────────────────────────────────────────────────── */
.sec-head {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 56px 0 32px;
    animation: fadeUp .8s cubic-bezier(.16,1,.3,1) both;
}
.sec-num {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #63b3ed, #4fd1c5);
    border-radius: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Space Mono', monospace;
    font-size: 14px;
    font-weight: 700;
    color: #03050f;
    flex-shrink: 0;
    box-shadow: 0 6px 20px rgba(99,179,237,0.25);
}
.sec-title {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #e2e8f0;
}
.sec-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(99,179,237,0.15), transparent);
}

/* ── FORM PANEL ─────────────────────────────────────────────────────────── */
.panel {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(99,179,237,0.08);
    border-radius: 22px;
    padding: 44px 44px 36px;
    margin-bottom: 4px;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    position: relative;
    overflow: hidden;
    transition: border-color .3s ease, background .3s ease;
    animation: fadeUp .7s cubic-bezier(.16,1,.3,1) both;
}
.panel::after {
    content:'';
    position:absolute;
    top:0; left:0; right:0;
    height:1px;
    background: linear-gradient(90deg, transparent, rgba(99,179,237,0.35), transparent);
}
.panel:hover {
    border-color: rgba(99,179,237,0.16);
    background: rgba(255,255,255,0.035);
}

/* ── STREAMLIT WIDGETS ──────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(10,15,35,0.7) !important;
    border: 1px solid rgba(99,179,237,0.15) !important;
    border-radius: 13px !important;
    color: #e2e8f0 !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    min-height: 48px !important;
    backdrop-filter: blur(12px) !important;
    transition: border-color .25s ease, box-shadow .25s ease !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: rgba(99,179,237,0.4) !important;
    box-shadow: 0 0 24px rgba(99,179,237,0.12) !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #4fd1c5 !important;
    box-shadow: 0 0 32px rgba(79,209,197,0.2) !important;
}
ul[role="listbox"] {
    background: rgba(8,12,30,0.97) !important;
    border: 1px solid rgba(99,179,237,0.18) !important;
    border-radius: 14px !important;
    box-shadow: 0 24px 60px rgba(0,0,0,0.6) !important;
    backdrop-filter: blur(20px) !important;
}
li[role="option"] {
    color: #cbd5e0 !important;
    font-size: 14px !important;
    font-family: 'Sora', sans-serif !important;
    padding: 11px 16px !important;
    transition: background .15s !important;
}
li[role="option"]:hover {
    background: rgba(99,179,237,0.12) !important;
    color: #63b3ed !important;
}
[data-testid="stWidgetLabel"] > div > p {
    color: #a0aec0 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: .4px !important;
    margin-bottom: 8px !important;
    font-family: 'Sora', sans-serif !important;
}
[data-testid="column"] { padding: 0 10px !important; }

/* ── BUTTON ─────────────────────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #2b6cb0 0%, #2c7a7b 100%) !important;
    color: white !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 19px 40px !important;
    border: 1px solid rgba(99,179,237,0.3) !important;
    border-radius: 15px !important;
    height: auto !important;
    margin-top: 28px !important;
    transition: all .3s cubic-bezier(.16,1,.3,1) !important;
    box-shadow: 0 16px 40px rgba(43,108,176,0.35) !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 24px 56px rgba(43,108,176,0.5), 0 0 60px rgba(79,209,197,0.15) !important;
    border-color: rgba(79,209,197,0.5) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(-1px) !important;
}

/* ── RESULT ─────────────────────────────────────────────────────────────── */
.result-wrap { animation: fadeUp .7s cubic-bezier(.16,1,.3,1); margin-top:52px; }
.result-box {
    border-radius: 26px;
    padding: 56px 44px;
    text-align: center;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
}
.result-box::after {
    content:'';
    position:absolute;
    top:0; left:0; right:0;
    height:1px;
    background: linear-gradient(90deg, transparent, var(--result-color), transparent);
    opacity: .6;
}
.result-ok  { background: rgba(104,211,145,0.05); border:1px solid rgba(104,211,145,0.2); --result-color:#68d391; }
.result-bad { background: rgba(252,129,129,0.05); border:1px solid rgba(252,129,129,0.2); --result-color:#fc8181; }
.result-icon { font-size:64px; display:block; margin-bottom:16px; animation:float 3s ease-in-out infinite; }
@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)} }
.result-status {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.result-ok  .result-status { color:#4fd1c5; }
.result-bad .result-status { color:#fc8181; }
.result-h2 { font-size:2rem; font-weight:800; color:#f0f4ff; margin-bottom:16px; letter-spacing:-1px; }
.result-conf {
    display:inline-block;
    font-family:'Space Mono',monospace;
    font-size:11px;
    font-weight:700;
    letter-spacing:1px;
    padding: 8px 18px;
    border-radius: 20px;
    margin-bottom:20px;
}
.result-ok  .result-conf { background:rgba(104,211,145,0.12); color:#68d391; border:1px solid rgba(104,211,145,0.25); }
.result-bad .result-conf { background:rgba(252,129,129,0.12); color:#fc8181; border:1px solid rgba(252,129,129,0.25); }
.result-desc { font-size:15px; line-height:1.75; color:#718096; max-width:460px; margin:0 auto 36px; font-weight:300; }
.bar-wrap { max-width:320px; margin:0 auto; }
.bar-labels { display:flex; justify-content:space-between; font-family:'Space Mono',monospace; font-size:10px; color:#4a5568; margin-bottom:8px; letter-spacing:.5px; }
.bar-track { height:5px; background:rgba(255,255,255,0.05); border-radius:3px; overflow:hidden; }
.bar-fill { height:100%; border-radius:3px; animation:grow 1s cubic-bezier(.16,1,.3,1); }
.result-ok  .bar-fill { background:linear-gradient(90deg,#68d391,#4fd1c5); box-shadow:0 0 14px rgba(104,211,145,.4); }
.result-bad .bar-fill { background:linear-gradient(90deg,#fc8181,#f6ad55); box-shadow:0 0 14px rgba(252,129,129,.4); }
@keyframes grow { from{width:0!important} }

/* ── RISK BREAKDOWN ─────────────────────────────────────────────────────── */
.risk-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin: 28px 0 0;
    text-align: left;
}
.risk-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 14px;
    padding: 16px;
}
.risk-card-label {
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #4a5568;
    margin-bottom: 6px;
}
.risk-card-val {
    font-family: 'Space Mono', monospace;
    font-size: 20px;
    font-weight: 700;
    background: linear-gradient(135deg, #63b3ed, #4fd1c5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── DISCLAIMER ─────────────────────────────────────────────────────────── */
.disclaimer {
    background: rgba(246,224,94,0.04);
    border: 1px solid rgba(246,224,94,0.12);
    border-radius: 14px;
    padding: 18px 20px;
    margin-top: 24px;
    display:flex; gap:12px; align-items:flex-start;
    font-size:13px; color:#b7a56a; line-height:1.7;
    backdrop-filter: blur(10px);
}

/* ── FOOTER ─────────────────────────────────────────────────────────────── */
.footer {
    text-align:center;
    margin-top:72px;
    padding-top:32px;
    border-top:1px solid rgba(99,179,237,0.07);
    font-family:'Space Mono',monospace;
    font-size:9px;
    letter-spacing:2px;
    text-transform:uppercase;
    color:#2d3748;
}

/* ── SPINNER ────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div { border-color:#63b3ed !important; }

/* ── RESPONSIVE ─────────────────────────────────────────────────────────── */
@media(max-width:720px){
    .shell { padding:48px 18px 80px; }
    .panel { padding:28px 18px; }
    .result-box { padding:40px 20px; }
    .hero-stats { gap:28px; }
    .hero-h1 { letter-spacing:-1.5px; }
    .risk-grid { grid-template-columns: 1fr 1fr; }
}
</style>
""", unsafe_allow_html=True)

# ── CANVAS NEURAL ANIMATION ──────────────────────────────────────────────────
st.markdown("""
<canvas id="mv-canvas" style="position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none;"></canvas>
<script>
(function(){
    var canvas = document.getElementById('mv-canvas');
    if(!canvas) return;
    var ctx = canvas.getContext('2d');

    function resize(){
        canvas.width  = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    var N   = 72;
    var MAXD = 190;
    var time = 0;

    function Neuron(){
        this.x  = Math.random() * canvas.width;
        this.y  = Math.random() * canvas.height;
        this.vx = (Math.random()-.5) * .45;
        this.vy = (Math.random()-.5) * .45;
        this.r  = 1.4 + Math.random() * 2.2;
        this.br = this.r;
        this.phase = Math.random() * Math.PI * 2;
        this.speed = .012 + Math.random() * .016;
        this.h  = 195 + Math.random() * 35;
        this.a  = .35 + Math.random() * .5;
    }
    Neuron.prototype.tick = function(){
        this.x += this.vx;
        this.y += this.vy;
        this.phase += this.speed;
        this.r = this.br + Math.sin(this.phase) * .9;
        if(this.x < 0 || this.x > canvas.width)  this.vx *= -1;
        if(this.y < 0 || this.y > canvas.height)  this.vy *= -1;
    };
    Neuron.prototype.draw = function(){
        var glow = .6 + Math.sin(this.phase) * .4;
        var g = ctx.createRadialGradient(this.x,this.y,0,this.x,this.y,this.r*6);
        g.addColorStop(0, 'hsla('+this.h+',85%,68%,'+(this.a*glow*.7)+')');
        g.addColorStop(1, 'hsla('+this.h+',85%,68%,0)');
        ctx.beginPath();
        ctx.arc(this.x,this.y,this.r*6,0,Math.PI*2);
        ctx.fillStyle = g;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(this.x,this.y,this.r,0,Math.PI*2);
        ctx.fillStyle = 'hsla('+this.h+',90%,78%,'+this.a+')';
        ctx.fill();
    };

    var neurons = [];
    for(var i=0;i<N;i++) neurons.push(new Neuron());

    function Signal(a,b){
        this.a = a; this.b = b;
        this.t = 0;
        this.spd = .018 + Math.random()*.018;
        this.h = 185 + Math.random()*25;
    }
    Signal.prototype.tick = function(){
        this.t += this.spd;
        return this.t >= 1;
    };
    Signal.prototype.draw = function(){
        var x = this.a.x + (this.b.x-this.a.x)*this.t;
        var y = this.a.y + (this.b.y-this.a.y)*this.t;
        var g = ctx.createRadialGradient(x,y,0,x,y,8);
        g.addColorStop(0,'hsla('+this.h+',100%,82%,.95)');
        g.addColorStop(.4,'hsla('+this.h+',100%,72%,.4)');
        g.addColorStop(1,'hsla('+this.h+',100%,72%,0)');
        ctx.beginPath();
        ctx.arc(x,y,8,0,Math.PI*2);
        ctx.fillStyle=g;
        ctx.fill();
    };

    var signals = [];
    var sigTimer = 0;

    function aurora(cx,cy,rx,ry,h,a,t){
        var x = cx + Math.sin(t*.7)  * rx * .35;
        var y = cy + Math.cos(t*.55) * ry * .3;
        var g = ctx.createRadialGradient(x,y,0,x,y,Math.max(rx,ry));
        g.addColorStop(0,'hsla('+h+',80%,55%,'+a+')');
        g.addColorStop(.5,'hsla('+(h+15)+',75%,50%,'+(a*.4)+')');
        g.addColorStop(1,'hsla('+h+',80%,55%,0)');
        ctx.beginPath();
        ctx.ellipse(x,y,rx,ry,t*.08,0,Math.PI*2);
        ctx.fillStyle=g;
        ctx.fill();
    }

    function ShootingStar(){ this.reset(); }
    ShootingStar.prototype.reset = function(){
        this.x  = Math.random() * canvas.width;
        this.y  = Math.random() * canvas.height * .5;
        this.len= 60 + Math.random()*80;
        this.spd= 4 + Math.random()*5;
        this.a  = .6 + Math.random()*.3;
        this.life = 0;
        this.maxLife = 40 + Math.random()*30;
        this.ang = Math.PI*.18 + Math.random()*.12;
    };
    ShootingStar.prototype.tick = function(){
        this.x += Math.cos(this.ang)*this.spd;
        this.y += Math.sin(this.ang)*this.spd;
        this.life++;
        if(this.life>this.maxLife || this.x>canvas.width+200 || this.y>canvas.height)
            this.reset();
    };
    ShootingStar.prototype.draw = function(){
        var p = this.life/this.maxLife;
        var a = this.a * Math.sin(p*Math.PI);
        var g = ctx.createLinearGradient(
            this.x,this.y,
            this.x-Math.cos(this.ang)*this.len,
            this.y-Math.sin(this.ang)*this.len
        );
        g.addColorStop(0,'rgba(180,220,255,'+a+')');
        g.addColorStop(1,'rgba(180,220,255,0)');
        ctx.beginPath();
        ctx.moveTo(this.x,this.y);
        ctx.lineTo(this.x-Math.cos(this.ang)*this.len, this.y-Math.sin(this.ang)*this.len);
        ctx.strokeStyle=g;
        ctx.lineWidth=1.5;
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(this.x,this.y,1.8,0,Math.PI*2);
        ctx.fillStyle='rgba(200,230,255,'+a+')';
        ctx.fill();
    };

    var stars=[];
    for(var s=0;s<4;s++) stars.push(new ShootingStar());
    stars.forEach(function(s,i){ s.life = Math.floor(s.maxLife*(i/4)); });

    function vignette(){
        var g = ctx.createRadialGradient(
            canvas.width*.5, canvas.height*.5, canvas.height*.15,
            canvas.width*.5, canvas.height*.5, canvas.width*.75
        );
        g.addColorStop(0,'rgba(3,5,15,0)');
        g.addColorStop(1,'rgba(3,5,15,.72)');
        ctx.fillStyle=g;
        ctx.fillRect(0,0,canvas.width,canvas.height);
    }

    function loop(){
        time += .007;
        sigTimer++;

        var bg=ctx.createLinearGradient(0,0,canvas.width,canvas.height);
        bg.addColorStop(0,'#03050f');
        bg.addColorStop(.5,'#070c1e');
        bg.addColorStop(1,'#04060f');
        ctx.fillStyle=bg;
        ctx.fillRect(0,0,canvas.width,canvas.height);

        ctx.globalCompositeOperation='screen';
        aurora(canvas.width*.18, canvas.height*.25, 320, 260, 215, .045, time);
        aurora(canvas.width*.82, canvas.height*.6,  280, 220, 195, .038, -time*.9);
        aurora(canvas.width*.5,  canvas.height*.75, 260, 200, 230, .030, time*1.1);
        aurora(canvas.width*.3,  canvas.height*.8,  200, 170, 170, .025, -time*.7);
        ctx.globalCompositeOperation='source-over';

        for(var i=0;i<neurons.length;i++){
            for(var j=i+1;j<neurons.length;j++){
                var dx=neurons[i].x-neurons[j].x;
                var dy=neurons[i].y-neurons[j].y;
                var d=Math.sqrt(dx*dx+dy*dy);
                if(d<MAXD){
                    var ratio=1-d/MAXD;
                    var h=200+ratio*30;
                    ctx.beginPath();
                    ctx.moveTo(neurons[i].x,neurons[i].y);
                    ctx.lineTo(neurons[j].x,neurons[j].y);
                    ctx.strokeStyle='hsla('+h+',75%,65%,'+(ratio*.22)+')';
                    ctx.lineWidth=ratio*1.6;
                    ctx.stroke();
                }
            }
        }

        if(sigTimer>45 && signals.length<12){
            sigTimer=0;
            var a=neurons[Math.floor(Math.random()*neurons.length)];
            var b=neurons[Math.floor(Math.random()*neurons.length)];
            if(a!==b){
                var ddx=a.x-b.x, ddy=a.y-b.y;
                if(Math.sqrt(ddx*ddx+ddy*ddy)<MAXD*1.4) signals.push(new Signal(a,b));
            }
        }

        for(var si=signals.length-1;si>=0;si--){
            signals[si].draw();
            if(signals[si].tick()) signals.splice(si,1);
        }

        neurons.forEach(function(n){ n.tick(); n.draw(); });

        stars.forEach(function(s){ s.draw(); s.tick(); });

        vignette();

        requestAnimationFrame(loop);
    }
    loop();
})();
</script>

<div class="shell">
""", unsafe_allow_html=True)

# ── NAVIGATION ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav">
    <div class="nav-logo">🧠 MindVault</div>
    <div class="nav-pill">
        <div class="nav-pill-dot"></div>
        Neural AI · Live
    </div>
</div>
""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">
        <div class="hero-tag-glow"></div>
        Mental Health Assessment
    </div>
    <div class="hero-h1">
        Your Mind<br><span>Matters Most</span>
    </div>
    <p class="hero-sub">
        14 thoughtful questions. One AI-powered insight.<br>
        Understand your mental wellness in under 60 seconds.
    </p>
    <div class="hero-stats">
        <div class="stat">
            <div class="stat-val">85%+</div>
            <div class="stat-lbl">Accuracy</div>
        </div>
        <div class="stat">
            <div class="stat-val">&lt;1s</div>
            <div class="stat-lbl">Result</div>
        </div>
        <div class="stat">
            <div class="stat-val">100%</div>
            <div class="stat-lbl">Private</div>
        </div>
    </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ── FORM MAPS ─────────────────────────────────────────────────────────────────
YESNO       = {"Yes": 1, "No": 0}
YESNO_MAYBE = {"Yes": 1, "Maybe": 0.5, "No": 0}
GENDER_MAP  = {"Female": 0, "Male": 1}
OCC_MAP     = {"Corporate": 0, "Self-Employed": 1, "Student": 2, "Other": 3}
DAYS_MAP    = {"Go out Every day": 0, "1–14 days": 1, "15–30 days": 2,
               "31–60 days": 3, "More than 2 months": 4}
MOOD_MAP    = {"Low": 0, "Medium": 1, "High": 2}
CARE_MAP    = {"No": 0, "Not sure": 0.5, "Yes": 1}

# ── SECTION 01 ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-head">
    <div class="sec-num">01</div>
    <div class="sec-title">Personal Background</div>
    <div class="sec-line"></div>
</div>
<div class="panel">
""", unsafe_allow_html=True)
c1, c2 = st.columns(2, gap="large")
with c1:
    gender     = st.selectbox("Gender",        list(GENDER_MAP.keys()), key="g")
    occupation = st.selectbox("Occupation",    list(OCC_MAP.keys()),    key="o")
with c2:
    self_employed  = st.selectbox("Self-employed?",                    list(YESNO.keys()), key="se")
    family_history = st.selectbox("Family history of mental illness?", list(YESNO.keys()), key="fh")
st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION 02 ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-head">
    <div class="sec-num">02</div>
    <div class="sec-title">Stress & Emotional State</div>
    <div class="sec-line"></div>
</div>
<div class="panel">
""", unsafe_allow_html=True)
c3, c4 = st.columns(2, gap="large")
with c3:
    growing_stress   = st.selectbox("Growing stress lately?",        list(YESNO_MAYBE.keys()), key="gs")
    mood_swings      = st.selectbox("Mood swing intensity?",         list(MOOD_MAP.keys()),    key="ms")
with c4:
    coping_struggles = st.selectbox("Struggling to cope daily?",     list(YESNO.keys()),       key="cs")
    changes_habits   = st.selectbox("Noticeable changes in habits?", list(YESNO_MAYBE.keys()), key="ch")
st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION 03 ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-head">
    <div class="sec-num">03</div>
    <div class="sec-title">Behaviour & Social Life</div>
    <div class="sec-line"></div>
</div>
<div class="panel">
""", unsafe_allow_html=True)
c5, c6 = st.columns(2, gap="large")
with c5:
    work_interest   = st.selectbox("Lost interest in work/hobbies?", list(YESNO_MAYBE.keys()), key="wi")
    social_weakness = st.selectbox("Feeling socially withdrawn?",    list(YESNO_MAYBE.keys()), key="sw")
with c6:
    days_indoors    = st.selectbox("Days spent indoors (past month)?", list(DAYS_MAP.keys()),  key="di")
st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION 04 ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-head">
    <div class="sec-num">04</div>
    <div class="sec-title">Awareness & Support Access</div>
    <div class="sec-line"></div>
</div>
<div class="panel">
""", unsafe_allow_html=True)
c7, c8 = st.columns(2, gap="large")
with c7:
    mh_history   = st.selectbox("Prior mental health episodes?",      list(YESNO_MAYBE.keys()), key="mhh")
    mh_interview = st.selectbox("Comfortable discussing MH at work?", list(YESNO_MAYBE.keys()), key="mhi")
with c8:
    care_options = st.selectbox("Access to mental health care?",      list(CARE_MAP.keys()),    key="co")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── RUN BUTTON ────────────────────────────────────────────────────────────────
predict_clicked = st.button("✦  Reveal My Assessment", key="predict_btn")

# ── PREDICTION ────────────────────────────────────────────────────────────────
if predict_clicked:
    with st.spinner("Analysing your responses…"):
        try:
            model = load_model()

            r = {
                "Gender":                  GENDER_MAP[gender],
                "Occupation":              OCC_MAP[occupation],
                "self_employed":           YESNO[self_employed],
                "family_history":          YESNO[family_history],
                "Days_Indoors":            DAYS_MAP[days_indoors],
                "Growing_Stress":          YESNO_MAYBE[growing_stress],
                "Changes_Habits":          YESNO_MAYBE[changes_habits],
                "Mental_Health_History":   YESNO_MAYBE[mh_history],
                "Mood_Swings":             MOOD_MAP[mood_swings],
                "Coping_Struggles":        YESNO[coping_struggles],
                "Work_Interest":           YESNO_MAYBE[work_interest],
                "Social_Weakness":         YESNO_MAYBE[social_weakness],
                "mental_health_interview": YESNO_MAYBE[mh_interview],
                "care_options":            CARE_MAP[care_options],
            }

            # ── Feature engineering (matches improved training script) ──────
            stress_score     = r["Growing_Stress"] + r["Mood_Swings"] + r["Coping_Struggles"]
            behavioral_score = r["Changes_Habits"] + r["Work_Interest"] + r["Social_Weakness"] + r["Days_Indoors"]
            awareness_score  = r["Mental_Health_History"] + r["mental_health_interview"] + r["care_options"]

            stress_x_family    = stress_score    * r["family_history"]
            care_x_family      = r["care_options"] * r["family_history"]
            awareness_x_family = awareness_score  * r["family_history"]
            gender_x_stress    = r["Gender"]      * stress_score

            # NEW engineered features (match train_improved_model.py)
            total_risk_score        = stress_score + behavioral_score
            mh_support_index        = r["care_options"] + r["mental_health_interview"]
            isolation_stress        = r["Days_Indoors"] * stress_score
            family_x_behav          = r["family_history"] * behavioral_score
            gender_x_family         = r["Gender"] * r["family_history"]
            care_x_stress           = r["care_options"] * stress_score
            coping_x_mood           = r["Coping_Struggles"] * r["Mood_Swings"]
            stress_awareness_ratio  = stress_score / (awareness_score + 0.5)

            high_risk_flag = int(
                stress_score     >= 3 and
                behavioral_score >= 4 and
                r["family_history"] == 1
            )
            very_high_risk = int(
                stress_score     >= 4 and
                behavioral_score >= 5 and
                r["family_history"] == 1 and
                r["care_options"]   <= 0.5
            )

            features = np.array([[
                r["Gender"], r["Occupation"], r["self_employed"], r["family_history"],
                r["Days_Indoors"], r["Growing_Stress"], r["Changes_Habits"],
                r["Mental_Health_History"], r["Mood_Swings"], r["Coping_Struggles"],
                r["Work_Interest"], r["Social_Weakness"], r["mental_health_interview"],
                r["care_options"],
                # original engineered
                stress_score, behavioral_score, awareness_score,
                stress_x_family, care_x_family, awareness_x_family,
                gender_x_stress, high_risk_flag,
                # new engineered
                total_risk_score, mh_support_index, isolation_stress,
                family_x_behav, gender_x_family, care_x_stress,
                coping_x_mood, stress_awareness_ratio, very_high_risk,
            ]])

            prediction = model.predict(features)[0]
            proba      = model.predict_proba(features)[0]
            confidence = float(max(proba))
            conf_pct   = round(confidence * 100, 1)
            conf_str   = f"{conf_pct}%"
            bar_w      = max(conf_pct, 5)
            is_bad     = int(prediction) == 1

            explanation = (
                "Based on your answers, seeking professional mental health support is recommended."
                if is_bad
                else "Your responses suggest a lower risk profile right now. Stay mindful!"
            )

            # Risk sub-scores for display (normalized 0-10)
            stress_disp   = round(min(stress_score / 4.0 * 10, 10), 1)
            behav_disp    = round(min(behavioral_score / 7.0 * 10, 10), 1)
            support_disp  = round(min(awareness_score / 3.0 * 10, 10), 1)

            risk_color = "result-bad" if is_bad else "result-ok"

            if is_bad:
                result_icon   = "🔴"
                result_status = "Assessment Complete"
                result_title  = "Support Recommended"
            else:
                result_icon   = "🟢"
                result_status = "Assessment Complete"
                result_title  = "Positive Mental State"

            st.markdown(f"""
            <div class="result-wrap">
              <div class="result-box {risk_color}">
                <span class="result-icon">{result_icon}</span>
                <div class="result-status">{result_status}</div>
                <div class="result-h2">{result_title}</div>
                <div class="result-conf">Confidence · {conf_str}</div>
                <p class="result-desc">{explanation}</p>
                <div class="bar-wrap">
                  <div class="bar-labels"><span>Confidence</span><span>{conf_str}</span></div>
                  <div class="bar-track"><div class="bar-fill" style="width:{bar_w}%"></div></div>
                </div>
                <div class="risk-grid">
                  <div class="risk-card">
                    <div class="risk-card-label">Stress Index</div>
                    <div class="risk-card-val">{stress_disp}<span style="font-size:12px;color:#4a5568">/10</span></div>
                  </div>
                  <div class="risk-card">
                    <div class="risk-card-label">Behavioural</div>
                    <div class="risk-card-val">{behav_disp}<span style="font-size:12px;color:#4a5568">/10</span></div>
                  </div>
                  <div class="risk-card">
                    <div class="risk-card-label">Support Access</div>
                    <div class="risk-card-val">{support_disp}<span style="font-size:12px;color:#4a5568">/10</span></div>
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="disclaimer">
                <span style="flex-shrink:0;margin-top:2px;">⚠️</span>
                <span><strong>Medical Disclaimer:</strong> MindVault is an educational ML demonstration —
                <strong>not a clinical diagnosis</strong>. If you are in distress, please consult a licensed
                mental health professional immediately.</span>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    MindVault &nbsp;·&nbsp; Powered by Streamlit &nbsp;·&nbsp; Not for Clinical Use
</div>
</div>
""", unsafe_allow_html=True)
