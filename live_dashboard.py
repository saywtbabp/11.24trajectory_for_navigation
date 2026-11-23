"""Live overview of the two routes compared by A* at the current doorway."""
from math import degrees
import tkinter as tk
from tkinter import ttk


PHASES = {
    "fly": "飞行", "approach": "接近门扇中点", "push": "推门", "boost": "加力试探",
    "release": "脱离并穿门", "retreat": "退离", "cross": "直接穿门",
    "bypass": "绕行", "no_path": "无可行路线",
}
SOURCES = {"range": "门扇测距", "range+imu": "IMU + 测距",
           "contact odometry": "机心轨迹估角", "held angle": "保持上次门角"}
ACTIONS = {"push": "推门", "cross": "直接穿门", "bypass": "绕行",
           "cross_after_push": "推开通过", "release": "脱离并穿门", "retreat": "退离后绕行",
           "inspect": "接近并观测", "unobserved": "绕行", "boost_start": "加力试探",
           "boost_end": "加力成功，恢复推门"}
REASONS = {"remaining_cost": "续推到终点的总代价高于退离绕行",
           "loaded_stall": "持续受载且门角没有足够进展",
           "contact_lost": "加力时失去有效接触",
           "angle_unobservable": "持续缺少可信门角", "push_timeout": "超过推门时限"}
DOOR_NAMES = ("短绕行", "已打开", "推门回波稀疏", "阻力上升", "中途限位",
              "高静阻力", "快速甩开", "关闭且门轴未知", "加力后打开")
METRICS = {"comparison", "precontact_update", "remaining"}
DOOR_COLOR = "#b75c1e"
AVOID_COLOR = "#764889"
FLIGHT_COLOR = "#166b7f"
TRAJECTORY_COLOR = "#218148"
MUTED = "#566d7a"


def number(value, unit=""):
    return "—" if value is None else f"{value:.2f}{unit}"


def explanation(snapshot, outcome):
    kind = outcome["event"] if outcome else snapshot.get("selected")
    reason = outcome.get("reason") if outcome else None
    if reason in REASONS:
        return REASONS[reason]
    if kind == "boost_start" or snapshot.get("decision_phase") == "boost":
        if snapshot.get("output_limited"):
            return "飞控输出已限幅。A* 继续用当前实测施力比较续推和绕行。"
        return "经门净收益仍为正，正在加力。A* 按当前估计施力更新续推代价。"
    if kind == "boost_end" and reason == "progress":
        return "加力后门角恢复进展，恢复普通推门并重新估计推行阻力。"
    if snapshot.get("continuation_blocked"):
        return "当前接触条件下，续推候选已移除。"
    if kind in {"cross", "release", "cross_after_push"}:
        return "当前几何满足机体净空要求，过门路线可行。"
    if kind == "unobserved":
        return "到达观测位置后仍缺少可信门扇几何，改走绕行口。"
    net = snapshot.get("net")
    if net is not None and kind in {"push", "bypass", "retreat"}:
        if abs(net) < .005:
            return "两条候选到任务终点的总代价近似相同，按全局 A* 选中的路线执行。"
        route = "经门" if net > 0 else "绕行"
        return f"{route}到任务终点的总代价较低，差值 {abs(net):.2f} 等效米。"
    if snapshot and snapshot.get("door_cost") is None:
        return "当前图中经门路线不可用。"
    return "等待下一次全局 A* 比较。"


class Dashboard:
    def __init__(self, arena, names):
        self.arena = arena
        self.names = dict(zip(names, DOOR_NAMES))
        self.closed = False
        self.event_index = 0
        self.path = []
        self.history = {name: [] for name in names}
        self.outcomes = {}
        self.focus = None
        self.next_draw = 0.
        self.map_size = (1000, 215)

        self.root = tk.Tk()
        self.root.title("无人机交互导航 · 实时全局 A* 决策")
        self.root.configure(padx=16, pady=12, bg="#edf2f5")
        self.root.minsize(900, 720)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.state = tk.StringVar(value="等待首次观测")
        ttk.Label(self.root, textvariable=self.state, padding=(0, 2, 0, 6)).pack(anchor="w")
        self.canvas = tk.Canvas(self.root, width=1000, height=215, bg="#f8fafb", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        legend = ttk.Frame(self.root, padding=(0, 4, 0, 6))
        legend.pack(fill="x")
        for label, color in (("已飞轨迹", FLIGHT_COLOR), ("局部优化轨迹", TRAJECTORY_COLOR),
                             ("经门候选", DOOR_COLOR),
                             ("绕行候选", AVOID_COLOR), ("门扇估计", "#555a35"),
                             ("空心点：比较起点", MUTED)):
            ttk.Label(legend, foreground=color, text=label, padding=(0, 0, 18, 0)).pack(side="left")

        details = ttk.LabelFrame(self.root, text="当前决策 · 到任务终点", padding=(10, 6))
        details.pack(fill="x")
        self.decision = tk.StringVar(value="等待首次全局比较")
        self.reason = tk.StringVar()
        self.signals = tk.StringVar()
        ttk.Label(details, textvariable=self.decision, font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w")
        ttk.Label(details, textvariable=self.reason, padding=(0, 3, 0, 4)).pack(anchor="w")
        self.costs = self._table(details, ("route", "length", "work", "scaled", "total"),
                                 ("候选", "几何长度 (m)", "剩余推门功 (J)",
                                  "功折算 (等效米)", "总代价 J (等效米)"),
                                 (120, 140, 155, 155, 170), 2)
        self.costs.insert("", "end", iid="door", values=("经门", "—", "—", "—", "—"))
        self.costs.insert("", "end", iid="avoid", values=("绕行", "—", "—", "—", "—"))
        self.costs.tag_configure("door", foreground=DOOR_COLOR)
        self.costs.tag_configure("avoid", foreground=AVOID_COLOR)
        ttk.Label(details, textvariable=self.signals, padding=(0, 5, 0, 1)).pack(anchor="w")

        ttk.Label(self.root, text="当前门的总代价走势", padding=(0, 5, 0, 2)).pack(anchor="w")
        self.trend = tk.Canvas(self.root, height=82, bg="#f8fafb", highlightthickness=0)
        self.trend.pack(fill="x")

        self.table = self._table(self.root, ("door", "fixture", "action", "door_cost", "avoid_cost", "net"),
                                 ("门", "仿真工况", "最新判断", "经门 J (等效米)",
                                  "绕行 J (等效米)", "净收益 C (等效米)"),
                                 (52, 138, 220, 155, 155, 158), len(names))
        for name in names:
            self.table.insert("", "end", iid=name, values=(name, self.names[name], "待观测", "—", "—", "—"))

    @staticmethod
    def _table(parent, keys, labels, widths, height):
        table = ttk.Treeview(parent, columns=keys, show="headings", selectmode="none", height=height)
        for key, label, width in zip(keys, labels, widths):
            table.heading(key, text=label, anchor="w")
            table.column(key, width=width, anchor="w")
        table.pack(fill="x")
        return table

    def xy(self, point):
        left, bottom, right, top = self.arena.bounds
        width, height = self.map_size
        scale = min((width-24)/(right-left), (height-18)/(top-bottom))
        return ((width-(right-left)*scale)/2 + (point[0]-left)*scale,
                (height-(top-bottom)*scale)/2 + (top-point[1])*scale)

    def close(self):
        if not self.closed:
            self.closed = True
            self.root.destroy()

    def pump(self):
        if not self.closed:
            self.root.update()

    def _line(self, path, color, width, dash=()):
        if len(path) > 1:
            self.canvas.create_line(*[value for point in path for value in self.xy(point)],
                                    fill=color, width=width, dash=dash)

    def _draw_map(self, position, target, nav, snapshot, trajectory=None):
        self.map_size = (max(100, self.canvas.winfo_width()), max(100, self.canvas.winfo_height()))
        self.canvas.delete("all")
        for gate in self.arena.gates:
            if gate.name == self.focus:
                x0, y0 = self.xy((gate.x-1.8, self.arena.bounds[3]))
                x1, y1 = self.xy((gate.x+1.8, self.arena.bounds[1]))
                self.canvas.create_rectangle(x0, y0, x1, y1, fill="#e6eef3", outline="")
        for x0, y0, x1, y1 in self.arena.walls:
            self._line(((x0, y0), (x1, y1)), "#83939d", 2)
        for gate in self.arena.gates:
            self._line(((gate.x, gate.low), (gate.x, gate.high)), "#c48136", 2, (3, 2))
            self.canvas.create_text(*self.xy((gate.x, gate.low-.65)), text=gate.name, fill="#304958")
            leaves = ((nav.attempt.leaf,) if nav.attempt and nav.active == gate.name
                      else nav.panels.get(gate.name, ()))
            for leaf in leaves:
                tip = leaf.hinge + gate.length*leaf.direction
                self._line((leaf.hinge, tip), "#555a35", 3, () if len(leaves) == 1 else (3, 2))
        self._line(self.path, FLIGHT_COLOR, 2)
        choices = (("door", DOOR_COLOR), ("avoid", AVOID_COLOR))
        decision_position = snapshot.get("decision_position") if snapshot else None
        if decision_position is not None:
            selected = "avoid" if snapshot.get("selected") in ("bypass", "retreat") else "door"
            for option, color in sorted(choices, key=lambda pair: pair[0] == selected):
                chosen = option == selected
                self._line(snapshot.get(option+"_route", ()), color, 3 if chosen else 2,
                           () if chosen else (5, 3))
        else:
            self._line(nav.route, "#98a8af", 2, (4, 3))
        if trajectory is not None:
            self._line(trajectory, TRAJECTORY_COLOR, 2, (2, 2))
        if decision_position is not None:
            x, y = self.xy(decision_position)
            self.canvas.create_oval(x-4, y-4, x+4, y+4, outline=MUTED, width=1)
        for point, color, radius in ((target, DOOR_COLOR, 3), (position, FLIGHT_COLOR, 5),
                                     (self.arena.goal, "#488051", 4)):
            x, y = self.xy(point)
            self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill=color, outline="")

    def _update_events(self, events):
        for event in events[self.event_index:]:
            name, kind = event.get("door"), event["event"]
            if name not in self.history:
                continue
            if kind in METRICS:
                self.focus = name
                self.history[name].append(event)
            if kind in ACTIONS and (kind != "boost_end" or event.get("reason") == "progress"):
                self.focus = name
                if kind != "bypass" or self.outcomes.get(name, {}).get("event") != "retreat":
                    self.outcomes[name] = event
            snapshot = self.history[name][-1] if self.history[name] else {}
            outcome = self.outcomes.get(name, {})
            if outcome.get("t", -1.) < snapshot.get("t", -1.):
                outcome = {}
            selected = "boost_start" if snapshot.get("decision_phase") == "boost" else snapshot.get("selected")
            action = ACTIONS.get(outcome.get("event", selected), "已观测")
            if outcome.get("reason") in REASONS:
                action = "退离 · " + {"remaining_cost": "续推代价较高", "loaded_stall": "受载停滞"}.get(
                    outcome["reason"], REASONS[outcome["reason"]])
            values = [number(snapshot.get(key)) for key in ("door_cost", "avoid_cost", "net")]
            self.table.item(name, values=(name, self.names[name], action, *values))
        self.event_index = len(events)

    def _show_decision(self, snapshot, outcome):
        if self.focus is None:
            return
        if outcome and outcome["t"] < snapshot.get("t", -1.):
            outcome = None
        selected = "boost_start" if snapshot.get("decision_phase") == "boost" else snapshot.get("selected")
        action = ACTIONS.get(outcome["event"] if outcome else selected, "待比较")
        stamp = snapshot.get("t")
        timing = "" if stamp is None else f"   ·   A* 比较于 {stamp:.2f} s"
        self.decision.set(f"{self.focus}  {self.names[self.focus]}   →   {action}{timing}")
        reason = explanation(snapshot, outcome)
        if snapshot.get("hinge_options") == 2 and action == ACTIONS["push"]:
            reason += "  门轴仍有两侧候选，使用中点接触。"
        self.reason.set(reason)
        for option, title in (("door", "经门"), ("avoid", "绕行")):
            cost = snapshot.get(option+"_cost")
            work = snapshot.get(option+"_route_work")
            total = number(cost) if cost is not None else ("不可用" if snapshot else "—")
            scaled = None if work is None else snapshot["effort_weight"]*work
            self.costs.item(option, tags=(option,), values=(title, number(snapshot.get("path_"+option)),
                            number(work), number(scaled), total))
        self.signals.set(
            f"G {number(snapshot.get('gain'), ' m')}     C {number(snapshot.get('net'), ' 等效米')}"
            f"     λ {number(snapshot.get('effort_weight'), ' m/J')}"
            f"     推行阻力估计 {number(snapshot.get('force'), ' N')}"
            f"     剩余接触弧长 {number(snapshot.get('remaining_arc'), ' m')}"
            f"     门角 {number(snapshot.get('angle_deg'), '°')}")

    def _draw_trend(self):
        self.trend.delete("all")
        if self.focus is None:
            return
        rows = self.history[self.focus]
        traces = [(color, [(row["t"], row[key]) for row in rows if row.get(key) is not None])
                  for color, key in ((DOOR_COLOR, "door_cost"), (AVOID_COLOR, "avoid_cost"))]
        points = [point for _, trace in traces for point in trace]
        if len(points) < 2:
            return
        left, bottom = 50, 64
        right = max(left+10, self.trend.winfo_width()-40)
        lo, hi = min(v for _, v in points), max(v for _, v in points)
        start, finish = points[0][0], max(t for t, _ in points)
        self.trend.create_line(left, 8, left, bottom, right, bottom, fill="#a4b1b8")
        for y, value in ((11, hi), (bottom, lo)):
            self.trend.create_text(left-7, y, text=f"{value:.1f}", anchor="e", fill=MUTED)
        self.trend.create_text(left, bottom+9, text=f"{start:.1f} s", anchor="nw", fill=MUTED)
        self.trend.create_text(right, bottom+9, text=f"{finish:.1f} s", anchor="ne", fill=MUTED)
        for color, trace in traces:
            coordinates = [(left+(t-start)/max(.1, finish-start)*(right-left),
                            bottom-(v-lo)/max(.2, hi-lo)*(bottom-10)) for t, v in trace]
            if len(coordinates) > 1:
                self.trend.create_line(*[v for p in coordinates for v in p], fill=color, width=2)

    def update(self, t, position, target, nav, *, force=False, trajectory=None, trajectory_status=""):
        if self.closed or (t < self.next_draw and not force):
            return
        self.next_draw = t+.10
        self.path.append(tuple(position))
        self._update_events(nav.events)
        history = self.history.get(self.focus, ())
        snapshot = history[-1] if history else {}
        self._show_decision(snapshot, self.outcomes.get(self.focus))
        self._draw_map(position, target, nav, snapshot, trajectory)
        self._draw_trend()
        status = f"仿真 {t:.1f} s   ·   执行中  {PHASES.get(nav.phase, nav.phase)}"
        if nav.attempt:
            attempt = nav.attempt
            status += (f"   ·   {nav.active}   估计门角 {degrees(attempt.leaf.angle):.1f}°"
                       f"   切向施力估计 {attempt.measured_force:.2f} N"
                       f"   {SOURCES.get(attempt.source, attempt.source)}")
        if nav.trial:
            status += f"   ·   施力目标 {nav.trial.target:.2f} N"
            if nav.trial.output_limited:
                status += "   ·   控制输出限幅"
        if trajectory_status:
            status += f"   ·   {trajectory_status}"
        self.state.set(status)
        self.pump()

    def finish(self, result):
        if not self.closed:
            status = "到达终点" if result["reached"] else result["reason"]
            self.state.set(f"{status}   ·   {result['seconds']:.2f} s   ·   {result['distance_m']:.2f} m"
                           f"   ·   墙面接触步数 {result['wall_contact_steps']}")
            self.pump()
