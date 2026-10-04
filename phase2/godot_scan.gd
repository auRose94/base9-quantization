extends SceneTree

# Loads every .gd under res:// in ONE Godot process and reports pass/fail per
# file. Calling `godot --check-only --script` per file costs ~0.3 s of process
# startup each; for a 200-file project that is a minute per project. Doing the
# same work inside a single run makes a full-corpus pass tractable.
#
# `load()` is NOT a validator: it returns a non-null GDScript resource even when
# compilation failed, printing only a SCRIPT ERROR. The reliable in-process
# signal is `reload()` — it returns ERR_PARSE_ERROR (43) for a script that does
# not compile. `can_instantiate()` agrees but would also reject legitimately
# non-instantiable scripts (abstract bases), so it is only reported, not judged.

const SKIP_DIRS := [".godot", ".git", ".vscode"]

var files: Array[String] = []


func _walk(dir_path: String) -> void:
	var d := DirAccess.open(dir_path)
	if d == null:
		return
	d.list_dir_begin()
	var n := d.get_next()
	while n != "":
		var skip := n.begins_with(".")
		if not skip and d.current_is_dir():
			for s in SKIP_DIRS:
				if n == s:
					skip = true
		var p := dir_path.path_join(n)
		if skip:
			pass
		elif d.current_is_dir():
			_walk(p)
		elif n.ends_with(".gd") and n != "godot_scan.gd":
			files.append(p)
		n = d.get_next()
	d.list_dir_end()


func _initialize() -> void:
	_walk("res://")
	files.sort()
	var bad := 0
	for f in files:
		var s = load(f)
		var err := 1
		if s != null and s is GDScript:
			err = s.reload()
		if err != OK:
			bad += 1
			print("FAIL ", err, " ", f)
		else:
			print("OK ", f)
	print("SUMMARY ", files.size() - bad, "/", files.size())
	quit()
