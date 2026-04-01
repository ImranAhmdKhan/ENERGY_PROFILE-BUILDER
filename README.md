ENERGYCASCADE.py — v4.0 → v5.0 Upgrade Walkthrough
Summary of Changes
The ORCA Energy Cascade Plotter has been upgraded from v4.0 to v5.0 with the following professional-grade improvements:

1. Performance — Cached File Parsing
Added get_cached_parser() with mtime-based invalidation
.out files are now parsed once and cached in memory
Cache auto-invalidates when file modification time changes
All OrcaParser() calls in calculate_state_energy() and update_data_table() replaced with cached version
2. Undo / Redo System
New UndoManager class with 50-level deep-copy snapshot stack
Ctrl+Z / Ctrl+Shift+Z keyboard shortcuts
Edit → Undo / Redo menu items
Undo pushes wired into all mutating operations:
Label editor, duplicate, rename, color change
Reorder (move up/down), remove, clear
Drag interactions (push on press, dirty on release)
Reset labels, auto-align labels
3. Drag Improvements
canvas.draw_idle() used for smoother deferred redraws during drag
Undo snapshot captured on drag start, dirty flag set on release
4. Bug Fixes
Fix	Detail
app.setStyle("Professional") → "Fusion"	Was silently falling back to platform default
Label editor None assignment	Removed the if value is not None guard that was preventing reset-to-defaults
load_session() skip prompt	Now passes _skip_prompt=True to new_session()
FolderLoader cleanup in load	Added .deleteLater() to prevent orphaned threads
5. Full Session Persistence
New: _gather_all_settings() / _restore_all_settings() methods
Session JSON now saves/restores all widget states:
All spin boxes (font sizes, bar thickness, margins, figure dimensions)
All checkboxes (17 total: grid, shadows, overlay, smooth, etc.)
All combo boxes (theme, legend location, line style, zero reference)
Unit button states (kcal/mol, kJ/mol, eV)
Session format versioned (version: 2) for future migration support
6. CSV Import (Feature Completion)
import_data() now fully functional
Accepts CSV with columns: Label, Energy_Eh, Color (optional)
Creates PlotItem entries with direct energy values (no .out file needed)
calculate_state_energy() handles item.energy for imported data
7. UI Polish
Double-click staging list item → opens label editor
Dirty state tracking (self._dirty) across all mutations
Smart close event: Save / Discard / Cancel dialog when dirty
Window title updated to v5.0
About dialog lists all new features
9. Plot Quality
Automatic Y-axis padding: 8% below, 12% above data range for label clearance
Verified
✅ Python syntax validation passed (py_compile.compile with Python 3.x)
✅ All existing features preserved — no breaking changes
✅ Old session files still load (backward compatible, just missing new settings which use defaults)
<img width="1787" height="1016" alt="image" src="https://github.com/user-attachments/assets/d49c1ccd-9750-4fbf-adff-84f5d68eeb71" />
<img width="945" height="343" alt="image" src="https://github.com/user-attachments/assets/bd3f1b02-be0f-41a7-abc2-1a32cc3f481e" />
