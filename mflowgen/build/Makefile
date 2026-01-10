#=========================================================================
# Makefile
#=========================================================================
# Generator : /home/sychae/cuda-vscode/mflowgen/mflowgen/backends/make_backend.py

SHELL=/usr/bin/env bash -euo pipefail

# $1 -- $dst
# $2 -- $src
# $3 -- $stamp

define cpdir
	rm -rf ./$1
	cp -aL $2 $1 || true
	chmod -R +w $1
	touch $3
endef

# $1 -- $dst
# $2 -- $src
# $3 -- $stamp

define cpdir-and-parameterize
	rm -rf ./$1
	cp -aL $2 $1 || true
	chmod -R +w $1
	cp .mflowgen/$1/configure.yml $1
	touch $3
endef

# $1 -- $dst
# $2 -- $src
# $3 -- $stamp

define mkdir-and-symlink
	rm -rf ./$1
	mkdir -p $1
	cd $1 && ln -sf ../$2/* . && cd ..
	rm $1/configure.yml && cp .mflowgen/$1/configure.yml $1
	touch $3
endef

# $1 -- $dst_dir
# $2 -- $dst
# $3 -- $src
# $4 -- $stamp

define symlink
	mkdir -p $1
	cd $1 && ln -sf $3 $2 && touch $4
endef

# $1 -- $stamp

define stamp
	touch $1
endef

define make-subgraph-dir
       mkdir -p $1
       chmod -R +w $1
       cd $1 && mflowgen run --subgraph --design $2
       touch $3
endef

# Default

default: info orfs-design orfs-docker-setup orfs-yosys-synthesis orfs-openroad-floorplan orfs-openroad-place orfs-openroad-cts orfs-openroad-route orfs-openroad-finish

# ------------------------------------------------------------------------
# info
# ------------------------------------------------------------------------

# build dir

ifeq ("$(wildcard 0-info/.prebuilt)","")
0-info/.stamp:
	$(call cpdir-and-parameterize,0-info,../nodes/info,0-info/.stamp)
endif

# collect inputs

# execute

ifeq ("$(wildcard 0-info/.prebuilt)","")

define 0_info_commands_rule
	mkdir -p 0-info/outputs && set -o pipefail && /home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-letters -c -t info && chmod +x .mflowgen/0-info/mflowgen-run && cp -f .mflowgen/0-info/mflowgen-run 0-info && if [[ -e .mflowgen/0-info/mflowgen-debug ]]; then chmod +x .mflowgen/0-info/mflowgen-debug && cp -f .mflowgen/0-info/mflowgen-debug 0-info; fi && if [[ -e .mflowgen/0-info/mflowgen-check-preconditions.py ]]; then chmod +x .mflowgen/0-info/mflowgen-check-preconditions.py && cp -f .mflowgen/0-info/mflowgen-check-preconditions.py 0-info; fi && if [[ -e .mflowgen/0-info/mflowgen-check-postconditions.py ]]; then chmod +x .mflowgen/0-info/mflowgen-check-postconditions.py && cp -f .mflowgen/0-info/mflowgen-check-postconditions.py 0-info; fi && cd 0-info && if [[ -e mflowgen-check-preconditions.py ]]; then ./mflowgen-check-preconditions.py || exit 1; fi && ./mflowgen-run 2>&1 | tee mflowgen-run.log || exit 1 && cd .. && touch -c 0-info/outputs/*
endef

0-info/.execstamp: 0-info/.stamp
	$(call 0_info_commands_rule)

0-info/.execstamp.execute-phony: 0-info/.execstamp
	touch $@

endif

# collect outputs

# post-conditions

define 0_info_post_conditions_commands_rule
	cd 0-info && if [[ -e mflowgen-check-postconditions.py ]]; then ./mflowgen-check-postconditions.py || exit 1; fi && cd ..
endef

0-info/.postconditions.stamp: 0-info/.execstamp 0-info/.execstamp.execute-phony
	$(call 0_info_post_conditions_commands_rule)
	touch $@

# alias

.PHONY: info

info: 0-info/.execstamp 0-info/.execstamp.execute-phony 0-info/.postconditions.stamp

.PHONY: 0

0: 0-info/.execstamp 0-info/.execstamp.execute-phony 0-info/.postconditions.stamp

# debug

# ------------------------------------------------------------------------
# orfs-design
# ------------------------------------------------------------------------

# build dir

ifeq ("$(wildcard 1-orfs-design/.prebuilt)","")
1-orfs-design/.stamp:
	$(call cpdir-and-parameterize,1-orfs-design,../designs/GcdUnit/orfs-design,1-orfs-design/.stamp)
endif

# collect inputs

# execute

ifeq ("$(wildcard 1-orfs-design/.prebuilt)","")

define 1_orfs_design_commands_rule
	mkdir -p 1-orfs-design/outputs && set -o pipefail && /home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-letters -c -t orfs-design && chmod +x .mflowgen/1-orfs-design/mflowgen-run && cp -f .mflowgen/1-orfs-design/mflowgen-run 1-orfs-design && if [[ -e .mflowgen/1-orfs-design/mflowgen-debug ]]; then chmod +x .mflowgen/1-orfs-design/mflowgen-debug && cp -f .mflowgen/1-orfs-design/mflowgen-debug 1-orfs-design; fi && if [[ -e .mflowgen/1-orfs-design/mflowgen-check-preconditions.py ]]; then chmod +x .mflowgen/1-orfs-design/mflowgen-check-preconditions.py && cp -f .mflowgen/1-orfs-design/mflowgen-check-preconditions.py 1-orfs-design; fi && if [[ -e .mflowgen/1-orfs-design/mflowgen-check-postconditions.py ]]; then chmod +x .mflowgen/1-orfs-design/mflowgen-check-postconditions.py && cp -f .mflowgen/1-orfs-design/mflowgen-check-postconditions.py 1-orfs-design; fi && cd 1-orfs-design && if [[ -e mflowgen-check-preconditions.py ]]; then ./mflowgen-check-preconditions.py || exit 1; fi && ./mflowgen-run 2>&1 | tee mflowgen-run.log || exit 1 && cd .. && touch -c 1-orfs-design/outputs/*
endef

1-orfs-design/.execstamp: 1-orfs-design/.stamp
	$(call 1_orfs_design_commands_rule)
	touch $@

1-orfs-design/outputs/.execstamp.orfs-design: 1-orfs-design/.execstamp
	touch $@

endif

# collect outputs

1-orfs-design/outputs/.stamp.orfs-design: 1-orfs-design/.execstamp 1-orfs-design/outputs/.execstamp.orfs-design
	$(call stamp,1-orfs-design/outputs/.stamp.orfs-design)

# post-conditions

define 1_orfs_design_post_conditions_commands_rule
	cd 1-orfs-design && if [[ -e mflowgen-check-postconditions.py ]]; then ./mflowgen-check-postconditions.py || exit 1; fi && cd ..
endef

1-orfs-design/.postconditions.stamp: 1-orfs-design/.execstamp 1-orfs-design/outputs/.execstamp.orfs-design 1-orfs-design/outputs/.stamp.orfs-design
	$(call 1_orfs_design_post_conditions_commands_rule)
	touch $@

# alias

.PHONY: orfs-design

orfs-design: 1-orfs-design/.execstamp 1-orfs-design/.postconditions.stamp 1-orfs-design/outputs/.execstamp.orfs-design 1-orfs-design/outputs/.stamp.orfs-design

.PHONY: 1

1: 1-orfs-design/.execstamp 1-orfs-design/.postconditions.stamp 1-orfs-design/outputs/.execstamp.orfs-design 1-orfs-design/outputs/.stamp.orfs-design

# debug

# ------------------------------------------------------------------------
# orfs-docker-setup
# ------------------------------------------------------------------------

# build dir

ifeq ("$(wildcard 2-orfs-docker-setup/.prebuilt)","")
2-orfs-docker-setup/.stamp:
	$(call cpdir-and-parameterize,2-orfs-docker-setup,../nodes/orfs-docker-setup,2-orfs-docker-setup/.stamp)
endif

# collect inputs

# execute

ifeq ("$(wildcard 2-orfs-docker-setup/.prebuilt)","")

define 2_orfs_docker_setup_commands_rule
	mkdir -p 2-orfs-docker-setup/outputs && set -o pipefail && /home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-letters -c -t orfs-docker-setup && chmod +x .mflowgen/2-orfs-docker-setup/mflowgen-run && cp -f .mflowgen/2-orfs-docker-setup/mflowgen-run 2-orfs-docker-setup && if [[ -e .mflowgen/2-orfs-docker-setup/mflowgen-debug ]]; then chmod +x .mflowgen/2-orfs-docker-setup/mflowgen-debug && cp -f .mflowgen/2-orfs-docker-setup/mflowgen-debug 2-orfs-docker-setup; fi && if [[ -e .mflowgen/2-orfs-docker-setup/mflowgen-check-preconditions.py ]]; then chmod +x .mflowgen/2-orfs-docker-setup/mflowgen-check-preconditions.py && cp -f .mflowgen/2-orfs-docker-setup/mflowgen-check-preconditions.py 2-orfs-docker-setup; fi && if [[ -e .mflowgen/2-orfs-docker-setup/mflowgen-check-postconditions.py ]]; then chmod +x .mflowgen/2-orfs-docker-setup/mflowgen-check-postconditions.py && cp -f .mflowgen/2-orfs-docker-setup/mflowgen-check-postconditions.py 2-orfs-docker-setup; fi && cd 2-orfs-docker-setup && if [[ -e mflowgen-check-preconditions.py ]]; then ./mflowgen-check-preconditions.py || exit 1; fi && ./mflowgen-run 2>&1 | tee mflowgen-run.log || exit 1 && cd .. && touch -c 2-orfs-docker-setup/outputs/*
endef

2-orfs-docker-setup/.execstamp: 2-orfs-docker-setup/.stamp
	$(call 2_orfs_docker_setup_commands_rule)
	touch $@

2-orfs-docker-setup/outputs/.execstamp.orfs-docker-image.tar: 2-orfs-docker-setup/.execstamp
	touch $@

endif

# collect outputs

2-orfs-docker-setup/outputs/.stamp.orfs-docker-image.tar: 2-orfs-docker-setup/.execstamp 2-orfs-docker-setup/outputs/.execstamp.orfs-docker-image.tar
	$(call stamp,2-orfs-docker-setup/outputs/.stamp.orfs-docker-image.tar)

# post-conditions

define 2_orfs_docker_setup_post_conditions_commands_rule
	cd 2-orfs-docker-setup && if [[ -e mflowgen-check-postconditions.py ]]; then ./mflowgen-check-postconditions.py || exit 1; fi && cd ..
endef

2-orfs-docker-setup/.postconditions.stamp: 2-orfs-docker-setup/.execstamp 2-orfs-docker-setup/outputs/.execstamp.orfs-docker-image.tar 2-orfs-docker-setup/outputs/.stamp.orfs-docker-image.tar
	$(call 2_orfs_docker_setup_post_conditions_commands_rule)
	touch $@

# alias

.PHONY: orfs-docker-setup

orfs-docker-setup: 2-orfs-docker-setup/.execstamp 2-orfs-docker-setup/.postconditions.stamp 2-orfs-docker-setup/outputs/.execstamp.orfs-docker-image.tar 2-orfs-docker-setup/outputs/.stamp.orfs-docker-image.tar

.PHONY: 2

2: 2-orfs-docker-setup/.execstamp 2-orfs-docker-setup/.postconditions.stamp 2-orfs-docker-setup/outputs/.execstamp.orfs-docker-image.tar 2-orfs-docker-setup/outputs/.stamp.orfs-docker-image.tar

# debug

# ------------------------------------------------------------------------
# orfs-yosys-synthesis
# ------------------------------------------------------------------------

# build dir

ifeq ("$(wildcard 3-orfs-yosys-synthesis/.prebuilt)","")
3-orfs-yosys-synthesis/.stamp: 1-orfs-design/.execstamp 1-orfs-design/.postconditions.stamp 1-orfs-design/outputs/.execstamp.orfs-design 1-orfs-design/outputs/.stamp.orfs-design
3-orfs-yosys-synthesis/.stamp: 2-orfs-docker-setup/.execstamp 2-orfs-docker-setup/.postconditions.stamp 2-orfs-docker-setup/outputs/.execstamp.orfs-docker-image.tar 2-orfs-docker-setup/outputs/.stamp.orfs-docker-image.tar
3-orfs-yosys-synthesis/.stamp:
	$(call cpdir-and-parameterize,3-orfs-yosys-synthesis,../nodes/orfs-yosys-synthesis,3-orfs-yosys-synthesis/.stamp)
endif

# collect inputs

ifeq ("$(wildcard 3-orfs-yosys-synthesis/.prebuilt)","")
3-orfs-yosys-synthesis/inputs/.stamp.orfs-design: 3-orfs-yosys-synthesis/.stamp 1-orfs-design/outputs/.stamp.orfs-design
	$(call symlink,3-orfs-yosys-synthesis/inputs,orfs-design,../../1-orfs-design/outputs/orfs-design,.stamp.orfs-design)
endif

ifeq ("$(wildcard 3-orfs-yosys-synthesis/.prebuilt)","")
3-orfs-yosys-synthesis/inputs/.stamp.orfs-docker-image.tar: 3-orfs-yosys-synthesis/.stamp 2-orfs-docker-setup/outputs/.stamp.orfs-docker-image.tar
	$(call symlink,3-orfs-yosys-synthesis/inputs,orfs-docker-image.tar,../../2-orfs-docker-setup/outputs/orfs-docker-image.tar,.stamp.orfs-docker-image.tar)
endif

# execute

ifeq ("$(wildcard 3-orfs-yosys-synthesis/.prebuilt)","")

define 3_orfs_yosys_synthesis_commands_rule
	mkdir -p 3-orfs-yosys-synthesis/outputs && set -o pipefail && /home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-letters -c -t orfs-yosys-synthesis && chmod +x .mflowgen/3-orfs-yosys-synthesis/mflowgen-run && cp -f .mflowgen/3-orfs-yosys-synthesis/mflowgen-run 3-orfs-yosys-synthesis && if [[ -e .mflowgen/3-orfs-yosys-synthesis/mflowgen-debug ]]; then chmod +x .mflowgen/3-orfs-yosys-synthesis/mflowgen-debug && cp -f .mflowgen/3-orfs-yosys-synthesis/mflowgen-debug 3-orfs-yosys-synthesis; fi && if [[ -e .mflowgen/3-orfs-yosys-synthesis/mflowgen-check-preconditions.py ]]; then chmod +x .mflowgen/3-orfs-yosys-synthesis/mflowgen-check-preconditions.py && cp -f .mflowgen/3-orfs-yosys-synthesis/mflowgen-check-preconditions.py 3-orfs-yosys-synthesis; fi && if [[ -e .mflowgen/3-orfs-yosys-synthesis/mflowgen-check-postconditions.py ]]; then chmod +x .mflowgen/3-orfs-yosys-synthesis/mflowgen-check-postconditions.py && cp -f .mflowgen/3-orfs-yosys-synthesis/mflowgen-check-postconditions.py 3-orfs-yosys-synthesis; fi && cd 3-orfs-yosys-synthesis && if [[ -e mflowgen-check-preconditions.py ]]; then ./mflowgen-check-preconditions.py || exit 1; fi && ./mflowgen-run 2>&1 | tee mflowgen-run.log || exit 1 && cd .. && touch -c 3-orfs-yosys-synthesis/outputs/*
endef

3-orfs-yosys-synthesis/.execstamp: 3-orfs-yosys-synthesis/.stamp 3-orfs-yosys-synthesis/inputs/.stamp.orfs-design 3-orfs-yosys-synthesis/inputs/.stamp.orfs-docker-image.tar
	$(call 3_orfs_yosys_synthesis_commands_rule)
	touch $@

3-orfs-yosys-synthesis/outputs/.execstamp.flow-checkpoint.tar.gz: 3-orfs-yosys-synthesis/.execstamp
	touch $@

endif

# collect outputs

3-orfs-yosys-synthesis/outputs/.stamp.flow-checkpoint.tar.gz: 3-orfs-yosys-synthesis/.execstamp 3-orfs-yosys-synthesis/outputs/.execstamp.flow-checkpoint.tar.gz
	$(call stamp,3-orfs-yosys-synthesis/outputs/.stamp.flow-checkpoint.tar.gz)

# post-conditions

define 3_orfs_yosys_synthesis_post_conditions_commands_rule
	cd 3-orfs-yosys-synthesis && if [[ -e mflowgen-check-postconditions.py ]]; then ./mflowgen-check-postconditions.py || exit 1; fi && cd ..
endef

3-orfs-yosys-synthesis/.postconditions.stamp: 3-orfs-yosys-synthesis/.execstamp 3-orfs-yosys-synthesis/outputs/.execstamp.flow-checkpoint.tar.gz 3-orfs-yosys-synthesis/outputs/.stamp.flow-checkpoint.tar.gz
	$(call 3_orfs_yosys_synthesis_post_conditions_commands_rule)
	touch $@

# alias

.PHONY: orfs-yosys-synthesis

orfs-yosys-synthesis: 3-orfs-yosys-synthesis/.execstamp 3-orfs-yosys-synthesis/.postconditions.stamp 3-orfs-yosys-synthesis/outputs/.execstamp.flow-checkpoint.tar.gz 3-orfs-yosys-synthesis/outputs/.stamp.flow-checkpoint.tar.gz

.PHONY: 3

3: 3-orfs-yosys-synthesis/.execstamp 3-orfs-yosys-synthesis/.postconditions.stamp 3-orfs-yosys-synthesis/outputs/.execstamp.flow-checkpoint.tar.gz 3-orfs-yosys-synthesis/outputs/.stamp.flow-checkpoint.tar.gz

# debug

define 3_debug_rule
	cd 3-orfs-yosys-synthesis && ./mflowgen-debug 2>&1 | tee mflowgen-debug.log
endef

debug-orfs-yosys-synthesis: 
	$(call 3_debug_rule)

.PHONY: debug-3

debug-3: debug-orfs-yosys-synthesis

# ------------------------------------------------------------------------
# orfs-openroad-floorplan
# ------------------------------------------------------------------------

# build dir

ifeq ("$(wildcard 4-orfs-openroad-floorplan/.prebuilt)","")
4-orfs-openroad-floorplan/.stamp: 2-orfs-docker-setup/.execstamp 2-orfs-docker-setup/.postconditions.stamp 2-orfs-docker-setup/outputs/.execstamp.orfs-docker-image.tar 2-orfs-docker-setup/outputs/.stamp.orfs-docker-image.tar
4-orfs-openroad-floorplan/.stamp: 3-orfs-yosys-synthesis/.execstamp 3-orfs-yosys-synthesis/.postconditions.stamp 3-orfs-yosys-synthesis/outputs/.execstamp.flow-checkpoint.tar.gz 3-orfs-yosys-synthesis/outputs/.stamp.flow-checkpoint.tar.gz
4-orfs-openroad-floorplan/.stamp:
	$(call cpdir-and-parameterize,4-orfs-openroad-floorplan,../nodes/orfs-openroad-floorplan,4-orfs-openroad-floorplan/.stamp)
endif

# collect inputs

ifeq ("$(wildcard 4-orfs-openroad-floorplan/.prebuilt)","")
4-orfs-openroad-floorplan/inputs/.stamp.flow-checkpoint.tar.gz: 4-orfs-openroad-floorplan/.stamp 3-orfs-yosys-synthesis/outputs/.stamp.flow-checkpoint.tar.gz
	$(call symlink,4-orfs-openroad-floorplan/inputs,flow-checkpoint.tar.gz,../../3-orfs-yosys-synthesis/outputs/flow-checkpoint.tar.gz,.stamp.flow-checkpoint.tar.gz)
endif

ifeq ("$(wildcard 4-orfs-openroad-floorplan/.prebuilt)","")
4-orfs-openroad-floorplan/inputs/.stamp.orfs-docker-image.tar: 4-orfs-openroad-floorplan/.stamp 2-orfs-docker-setup/outputs/.stamp.orfs-docker-image.tar
	$(call symlink,4-orfs-openroad-floorplan/inputs,orfs-docker-image.tar,../../2-orfs-docker-setup/outputs/orfs-docker-image.tar,.stamp.orfs-docker-image.tar)
endif

# execute

ifeq ("$(wildcard 4-orfs-openroad-floorplan/.prebuilt)","")

define 4_orfs_openroad_floorplan_commands_rule
	mkdir -p 4-orfs-openroad-floorplan/outputs && set -o pipefail && /home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-letters -c -t orfs-openroad-floorplan && chmod +x .mflowgen/4-orfs-openroad-floorplan/mflowgen-run && cp -f .mflowgen/4-orfs-openroad-floorplan/mflowgen-run 4-orfs-openroad-floorplan && if [[ -e .mflowgen/4-orfs-openroad-floorplan/mflowgen-debug ]]; then chmod +x .mflowgen/4-orfs-openroad-floorplan/mflowgen-debug && cp -f .mflowgen/4-orfs-openroad-floorplan/mflowgen-debug 4-orfs-openroad-floorplan; fi && if [[ -e .mflowgen/4-orfs-openroad-floorplan/mflowgen-check-preconditions.py ]]; then chmod +x .mflowgen/4-orfs-openroad-floorplan/mflowgen-check-preconditions.py && cp -f .mflowgen/4-orfs-openroad-floorplan/mflowgen-check-preconditions.py 4-orfs-openroad-floorplan; fi && if [[ -e .mflowgen/4-orfs-openroad-floorplan/mflowgen-check-postconditions.py ]]; then chmod +x .mflowgen/4-orfs-openroad-floorplan/mflowgen-check-postconditions.py && cp -f .mflowgen/4-orfs-openroad-floorplan/mflowgen-check-postconditions.py 4-orfs-openroad-floorplan; fi && cd 4-orfs-openroad-floorplan && if [[ -e mflowgen-check-preconditions.py ]]; then ./mflowgen-check-preconditions.py || exit 1; fi && ./mflowgen-run 2>&1 | tee mflowgen-run.log || exit 1 && cd .. && touch -c 4-orfs-openroad-floorplan/outputs/*
endef

4-orfs-openroad-floorplan/.execstamp: 4-orfs-openroad-floorplan/.stamp 4-orfs-openroad-floorplan/inputs/.stamp.flow-checkpoint.tar.gz 4-orfs-openroad-floorplan/inputs/.stamp.orfs-docker-image.tar
	$(call 4_orfs_openroad_floorplan_commands_rule)
	touch $@

4-orfs-openroad-floorplan/outputs/.execstamp.flow-checkpoint.tar.gz: 4-orfs-openroad-floorplan/.execstamp
	touch $@

endif

# collect outputs

4-orfs-openroad-floorplan/outputs/.stamp.flow-checkpoint.tar.gz: 4-orfs-openroad-floorplan/.execstamp 4-orfs-openroad-floorplan/outputs/.execstamp.flow-checkpoint.tar.gz
	$(call stamp,4-orfs-openroad-floorplan/outputs/.stamp.flow-checkpoint.tar.gz)

# post-conditions

define 4_orfs_openroad_floorplan_post_conditions_commands_rule
	cd 4-orfs-openroad-floorplan && if [[ -e mflowgen-check-postconditions.py ]]; then ./mflowgen-check-postconditions.py || exit 1; fi && cd ..
endef

4-orfs-openroad-floorplan/.postconditions.stamp: 4-orfs-openroad-floorplan/.execstamp 4-orfs-openroad-floorplan/outputs/.execstamp.flow-checkpoint.tar.gz 4-orfs-openroad-floorplan/outputs/.stamp.flow-checkpoint.tar.gz
	$(call 4_orfs_openroad_floorplan_post_conditions_commands_rule)
	touch $@

# alias

.PHONY: orfs-openroad-floorplan

orfs-openroad-floorplan: 4-orfs-openroad-floorplan/.execstamp 4-orfs-openroad-floorplan/.postconditions.stamp 4-orfs-openroad-floorplan/outputs/.execstamp.flow-checkpoint.tar.gz 4-orfs-openroad-floorplan/outputs/.stamp.flow-checkpoint.tar.gz

.PHONY: 4

4: 4-orfs-openroad-floorplan/.execstamp 4-orfs-openroad-floorplan/.postconditions.stamp 4-orfs-openroad-floorplan/outputs/.execstamp.flow-checkpoint.tar.gz 4-orfs-openroad-floorplan/outputs/.stamp.flow-checkpoint.tar.gz

# debug

define 4_debug_rule
	cd 4-orfs-openroad-floorplan && ./mflowgen-debug 2>&1 | tee mflowgen-debug.log
endef

debug-orfs-openroad-floorplan: 
	$(call 4_debug_rule)

.PHONY: debug-4

debug-4: debug-orfs-openroad-floorplan

# ------------------------------------------------------------------------
# orfs-openroad-place
# ------------------------------------------------------------------------

# build dir

ifeq ("$(wildcard 5-orfs-openroad-place/.prebuilt)","")
5-orfs-openroad-place/.stamp: 2-orfs-docker-setup/.execstamp 2-orfs-docker-setup/.postconditions.stamp 2-orfs-docker-setup/outputs/.execstamp.orfs-docker-image.tar 2-orfs-docker-setup/outputs/.stamp.orfs-docker-image.tar
5-orfs-openroad-place/.stamp: 4-orfs-openroad-floorplan/.execstamp 4-orfs-openroad-floorplan/.postconditions.stamp 4-orfs-openroad-floorplan/outputs/.execstamp.flow-checkpoint.tar.gz 4-orfs-openroad-floorplan/outputs/.stamp.flow-checkpoint.tar.gz
5-orfs-openroad-place/.stamp:
	$(call cpdir-and-parameterize,5-orfs-openroad-place,../nodes/orfs-openroad-place,5-orfs-openroad-place/.stamp)
endif

# collect inputs

ifeq ("$(wildcard 5-orfs-openroad-place/.prebuilt)","")
5-orfs-openroad-place/inputs/.stamp.flow-checkpoint.tar.gz: 5-orfs-openroad-place/.stamp 4-orfs-openroad-floorplan/outputs/.stamp.flow-checkpoint.tar.gz
	$(call symlink,5-orfs-openroad-place/inputs,flow-checkpoint.tar.gz,../../4-orfs-openroad-floorplan/outputs/flow-checkpoint.tar.gz,.stamp.flow-checkpoint.tar.gz)
endif

ifeq ("$(wildcard 5-orfs-openroad-place/.prebuilt)","")
5-orfs-openroad-place/inputs/.stamp.orfs-docker-image.tar: 5-orfs-openroad-place/.stamp 2-orfs-docker-setup/outputs/.stamp.orfs-docker-image.tar
	$(call symlink,5-orfs-openroad-place/inputs,orfs-docker-image.tar,../../2-orfs-docker-setup/outputs/orfs-docker-image.tar,.stamp.orfs-docker-image.tar)
endif

# execute

ifeq ("$(wildcard 5-orfs-openroad-place/.prebuilt)","")

define 5_orfs_openroad_place_commands_rule
	mkdir -p 5-orfs-openroad-place/outputs && set -o pipefail && /home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-letters -c -t orfs-openroad-place && chmod +x .mflowgen/5-orfs-openroad-place/mflowgen-run && cp -f .mflowgen/5-orfs-openroad-place/mflowgen-run 5-orfs-openroad-place && if [[ -e .mflowgen/5-orfs-openroad-place/mflowgen-debug ]]; then chmod +x .mflowgen/5-orfs-openroad-place/mflowgen-debug && cp -f .mflowgen/5-orfs-openroad-place/mflowgen-debug 5-orfs-openroad-place; fi && if [[ -e .mflowgen/5-orfs-openroad-place/mflowgen-check-preconditions.py ]]; then chmod +x .mflowgen/5-orfs-openroad-place/mflowgen-check-preconditions.py && cp -f .mflowgen/5-orfs-openroad-place/mflowgen-check-preconditions.py 5-orfs-openroad-place; fi && if [[ -e .mflowgen/5-orfs-openroad-place/mflowgen-check-postconditions.py ]]; then chmod +x .mflowgen/5-orfs-openroad-place/mflowgen-check-postconditions.py && cp -f .mflowgen/5-orfs-openroad-place/mflowgen-check-postconditions.py 5-orfs-openroad-place; fi && cd 5-orfs-openroad-place && if [[ -e mflowgen-check-preconditions.py ]]; then ./mflowgen-check-preconditions.py || exit 1; fi && ./mflowgen-run 2>&1 | tee mflowgen-run.log || exit 1 && cd .. && touch -c 5-orfs-openroad-place/outputs/*
endef

5-orfs-openroad-place/.execstamp: 5-orfs-openroad-place/.stamp 5-orfs-openroad-place/inputs/.stamp.flow-checkpoint.tar.gz 5-orfs-openroad-place/inputs/.stamp.orfs-docker-image.tar
	$(call 5_orfs_openroad_place_commands_rule)
	touch $@

5-orfs-openroad-place/outputs/.execstamp.flow-checkpoint.tar.gz: 5-orfs-openroad-place/.execstamp
	touch $@

endif

# collect outputs

5-orfs-openroad-place/outputs/.stamp.flow-checkpoint.tar.gz: 5-orfs-openroad-place/.execstamp 5-orfs-openroad-place/outputs/.execstamp.flow-checkpoint.tar.gz
	$(call stamp,5-orfs-openroad-place/outputs/.stamp.flow-checkpoint.tar.gz)

# post-conditions

define 5_orfs_openroad_place_post_conditions_commands_rule
	cd 5-orfs-openroad-place && if [[ -e mflowgen-check-postconditions.py ]]; then ./mflowgen-check-postconditions.py || exit 1; fi && cd ..
endef

5-orfs-openroad-place/.postconditions.stamp: 5-orfs-openroad-place/.execstamp 5-orfs-openroad-place/outputs/.execstamp.flow-checkpoint.tar.gz 5-orfs-openroad-place/outputs/.stamp.flow-checkpoint.tar.gz
	$(call 5_orfs_openroad_place_post_conditions_commands_rule)
	touch $@

# alias

.PHONY: orfs-openroad-place

orfs-openroad-place: 5-orfs-openroad-place/.execstamp 5-orfs-openroad-place/.postconditions.stamp 5-orfs-openroad-place/outputs/.execstamp.flow-checkpoint.tar.gz 5-orfs-openroad-place/outputs/.stamp.flow-checkpoint.tar.gz

.PHONY: 5

5: 5-orfs-openroad-place/.execstamp 5-orfs-openroad-place/.postconditions.stamp 5-orfs-openroad-place/outputs/.execstamp.flow-checkpoint.tar.gz 5-orfs-openroad-place/outputs/.stamp.flow-checkpoint.tar.gz

# debug

define 5_debug_rule
	cd 5-orfs-openroad-place && ./mflowgen-debug 2>&1 | tee mflowgen-debug.log
endef

debug-orfs-openroad-place: 
	$(call 5_debug_rule)

.PHONY: debug-5

debug-5: debug-orfs-openroad-place

# ------------------------------------------------------------------------
# orfs-openroad-cts
# ------------------------------------------------------------------------

# build dir

ifeq ("$(wildcard 6-orfs-openroad-cts/.prebuilt)","")
6-orfs-openroad-cts/.stamp: 2-orfs-docker-setup/.execstamp 2-orfs-docker-setup/.postconditions.stamp 2-orfs-docker-setup/outputs/.execstamp.orfs-docker-image.tar 2-orfs-docker-setup/outputs/.stamp.orfs-docker-image.tar
6-orfs-openroad-cts/.stamp: 5-orfs-openroad-place/.execstamp 5-orfs-openroad-place/.postconditions.stamp 5-orfs-openroad-place/outputs/.execstamp.flow-checkpoint.tar.gz 5-orfs-openroad-place/outputs/.stamp.flow-checkpoint.tar.gz
6-orfs-openroad-cts/.stamp:
	$(call cpdir-and-parameterize,6-orfs-openroad-cts,../nodes/orfs-openroad-cts,6-orfs-openroad-cts/.stamp)
endif

# collect inputs

ifeq ("$(wildcard 6-orfs-openroad-cts/.prebuilt)","")
6-orfs-openroad-cts/inputs/.stamp.flow-checkpoint.tar.gz: 6-orfs-openroad-cts/.stamp 5-orfs-openroad-place/outputs/.stamp.flow-checkpoint.tar.gz
	$(call symlink,6-orfs-openroad-cts/inputs,flow-checkpoint.tar.gz,../../5-orfs-openroad-place/outputs/flow-checkpoint.tar.gz,.stamp.flow-checkpoint.tar.gz)
endif

ifeq ("$(wildcard 6-orfs-openroad-cts/.prebuilt)","")
6-orfs-openroad-cts/inputs/.stamp.orfs-docker-image.tar: 6-orfs-openroad-cts/.stamp 2-orfs-docker-setup/outputs/.stamp.orfs-docker-image.tar
	$(call symlink,6-orfs-openroad-cts/inputs,orfs-docker-image.tar,../../2-orfs-docker-setup/outputs/orfs-docker-image.tar,.stamp.orfs-docker-image.tar)
endif

# execute

ifeq ("$(wildcard 6-orfs-openroad-cts/.prebuilt)","")

define 6_orfs_openroad_cts_commands_rule
	mkdir -p 6-orfs-openroad-cts/outputs && set -o pipefail && /home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-letters -c -t orfs-openroad-cts && chmod +x .mflowgen/6-orfs-openroad-cts/mflowgen-run && cp -f .mflowgen/6-orfs-openroad-cts/mflowgen-run 6-orfs-openroad-cts && if [[ -e .mflowgen/6-orfs-openroad-cts/mflowgen-debug ]]; then chmod +x .mflowgen/6-orfs-openroad-cts/mflowgen-debug && cp -f .mflowgen/6-orfs-openroad-cts/mflowgen-debug 6-orfs-openroad-cts; fi && if [[ -e .mflowgen/6-orfs-openroad-cts/mflowgen-check-preconditions.py ]]; then chmod +x .mflowgen/6-orfs-openroad-cts/mflowgen-check-preconditions.py && cp -f .mflowgen/6-orfs-openroad-cts/mflowgen-check-preconditions.py 6-orfs-openroad-cts; fi && if [[ -e .mflowgen/6-orfs-openroad-cts/mflowgen-check-postconditions.py ]]; then chmod +x .mflowgen/6-orfs-openroad-cts/mflowgen-check-postconditions.py && cp -f .mflowgen/6-orfs-openroad-cts/mflowgen-check-postconditions.py 6-orfs-openroad-cts; fi && cd 6-orfs-openroad-cts && if [[ -e mflowgen-check-preconditions.py ]]; then ./mflowgen-check-preconditions.py || exit 1; fi && ./mflowgen-run 2>&1 | tee mflowgen-run.log || exit 1 && cd .. && touch -c 6-orfs-openroad-cts/outputs/*
endef

6-orfs-openroad-cts/.execstamp: 6-orfs-openroad-cts/.stamp 6-orfs-openroad-cts/inputs/.stamp.flow-checkpoint.tar.gz 6-orfs-openroad-cts/inputs/.stamp.orfs-docker-image.tar
	$(call 6_orfs_openroad_cts_commands_rule)
	touch $@

6-orfs-openroad-cts/outputs/.execstamp.flow-checkpoint.tar.gz: 6-orfs-openroad-cts/.execstamp
	touch $@

endif

# collect outputs

6-orfs-openroad-cts/outputs/.stamp.flow-checkpoint.tar.gz: 6-orfs-openroad-cts/.execstamp 6-orfs-openroad-cts/outputs/.execstamp.flow-checkpoint.tar.gz
	$(call stamp,6-orfs-openroad-cts/outputs/.stamp.flow-checkpoint.tar.gz)

# post-conditions

define 6_orfs_openroad_cts_post_conditions_commands_rule
	cd 6-orfs-openroad-cts && if [[ -e mflowgen-check-postconditions.py ]]; then ./mflowgen-check-postconditions.py || exit 1; fi && cd ..
endef

6-orfs-openroad-cts/.postconditions.stamp: 6-orfs-openroad-cts/.execstamp 6-orfs-openroad-cts/outputs/.execstamp.flow-checkpoint.tar.gz 6-orfs-openroad-cts/outputs/.stamp.flow-checkpoint.tar.gz
	$(call 6_orfs_openroad_cts_post_conditions_commands_rule)
	touch $@

# alias

.PHONY: orfs-openroad-cts

orfs-openroad-cts: 6-orfs-openroad-cts/.execstamp 6-orfs-openroad-cts/.postconditions.stamp 6-orfs-openroad-cts/outputs/.execstamp.flow-checkpoint.tar.gz 6-orfs-openroad-cts/outputs/.stamp.flow-checkpoint.tar.gz

.PHONY: 6

6: 6-orfs-openroad-cts/.execstamp 6-orfs-openroad-cts/.postconditions.stamp 6-orfs-openroad-cts/outputs/.execstamp.flow-checkpoint.tar.gz 6-orfs-openroad-cts/outputs/.stamp.flow-checkpoint.tar.gz

# debug

define 6_debug_rule
	cd 6-orfs-openroad-cts && ./mflowgen-debug 2>&1 | tee mflowgen-debug.log
endef

debug-orfs-openroad-cts: 
	$(call 6_debug_rule)

.PHONY: debug-6

debug-6: debug-orfs-openroad-cts

# ------------------------------------------------------------------------
# orfs-openroad-route
# ------------------------------------------------------------------------

# build dir

ifeq ("$(wildcard 7-orfs-openroad-route/.prebuilt)","")
7-orfs-openroad-route/.stamp: 2-orfs-docker-setup/.execstamp 2-orfs-docker-setup/.postconditions.stamp 2-orfs-docker-setup/outputs/.execstamp.orfs-docker-image.tar 2-orfs-docker-setup/outputs/.stamp.orfs-docker-image.tar
7-orfs-openroad-route/.stamp: 6-orfs-openroad-cts/.execstamp 6-orfs-openroad-cts/.postconditions.stamp 6-orfs-openroad-cts/outputs/.execstamp.flow-checkpoint.tar.gz 6-orfs-openroad-cts/outputs/.stamp.flow-checkpoint.tar.gz
7-orfs-openroad-route/.stamp:
	$(call cpdir-and-parameterize,7-orfs-openroad-route,../nodes/orfs-openroad-route,7-orfs-openroad-route/.stamp)
endif

# collect inputs

ifeq ("$(wildcard 7-orfs-openroad-route/.prebuilt)","")
7-orfs-openroad-route/inputs/.stamp.flow-checkpoint.tar.gz: 7-orfs-openroad-route/.stamp 6-orfs-openroad-cts/outputs/.stamp.flow-checkpoint.tar.gz
	$(call symlink,7-orfs-openroad-route/inputs,flow-checkpoint.tar.gz,../../6-orfs-openroad-cts/outputs/flow-checkpoint.tar.gz,.stamp.flow-checkpoint.tar.gz)
endif

ifeq ("$(wildcard 7-orfs-openroad-route/.prebuilt)","")
7-orfs-openroad-route/inputs/.stamp.orfs-docker-image.tar: 7-orfs-openroad-route/.stamp 2-orfs-docker-setup/outputs/.stamp.orfs-docker-image.tar
	$(call symlink,7-orfs-openroad-route/inputs,orfs-docker-image.tar,../../2-orfs-docker-setup/outputs/orfs-docker-image.tar,.stamp.orfs-docker-image.tar)
endif

# execute

ifeq ("$(wildcard 7-orfs-openroad-route/.prebuilt)","")

define 7_orfs_openroad_route_commands_rule
	mkdir -p 7-orfs-openroad-route/outputs && set -o pipefail && /home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-letters -c -t orfs-openroad-route && chmod +x .mflowgen/7-orfs-openroad-route/mflowgen-run && cp -f .mflowgen/7-orfs-openroad-route/mflowgen-run 7-orfs-openroad-route && if [[ -e .mflowgen/7-orfs-openroad-route/mflowgen-debug ]]; then chmod +x .mflowgen/7-orfs-openroad-route/mflowgen-debug && cp -f .mflowgen/7-orfs-openroad-route/mflowgen-debug 7-orfs-openroad-route; fi && if [[ -e .mflowgen/7-orfs-openroad-route/mflowgen-check-preconditions.py ]]; then chmod +x .mflowgen/7-orfs-openroad-route/mflowgen-check-preconditions.py && cp -f .mflowgen/7-orfs-openroad-route/mflowgen-check-preconditions.py 7-orfs-openroad-route; fi && if [[ -e .mflowgen/7-orfs-openroad-route/mflowgen-check-postconditions.py ]]; then chmod +x .mflowgen/7-orfs-openroad-route/mflowgen-check-postconditions.py && cp -f .mflowgen/7-orfs-openroad-route/mflowgen-check-postconditions.py 7-orfs-openroad-route; fi && cd 7-orfs-openroad-route && if [[ -e mflowgen-check-preconditions.py ]]; then ./mflowgen-check-preconditions.py || exit 1; fi && ./mflowgen-run 2>&1 | tee mflowgen-run.log || exit 1 && cd .. && touch -c 7-orfs-openroad-route/outputs/*
endef

7-orfs-openroad-route/.execstamp: 7-orfs-openroad-route/.stamp 7-orfs-openroad-route/inputs/.stamp.flow-checkpoint.tar.gz 7-orfs-openroad-route/inputs/.stamp.orfs-docker-image.tar
	$(call 7_orfs_openroad_route_commands_rule)
	touch $@

7-orfs-openroad-route/outputs/.execstamp.flow-checkpoint.tar.gz: 7-orfs-openroad-route/.execstamp
	touch $@

endif

# collect outputs

7-orfs-openroad-route/outputs/.stamp.flow-checkpoint.tar.gz: 7-orfs-openroad-route/.execstamp 7-orfs-openroad-route/outputs/.execstamp.flow-checkpoint.tar.gz
	$(call stamp,7-orfs-openroad-route/outputs/.stamp.flow-checkpoint.tar.gz)

# post-conditions

define 7_orfs_openroad_route_post_conditions_commands_rule
	cd 7-orfs-openroad-route && if [[ -e mflowgen-check-postconditions.py ]]; then ./mflowgen-check-postconditions.py || exit 1; fi && cd ..
endef

7-orfs-openroad-route/.postconditions.stamp: 7-orfs-openroad-route/.execstamp 7-orfs-openroad-route/outputs/.execstamp.flow-checkpoint.tar.gz 7-orfs-openroad-route/outputs/.stamp.flow-checkpoint.tar.gz
	$(call 7_orfs_openroad_route_post_conditions_commands_rule)
	touch $@

# alias

.PHONY: orfs-openroad-route

orfs-openroad-route: 7-orfs-openroad-route/.execstamp 7-orfs-openroad-route/.postconditions.stamp 7-orfs-openroad-route/outputs/.execstamp.flow-checkpoint.tar.gz 7-orfs-openroad-route/outputs/.stamp.flow-checkpoint.tar.gz

.PHONY: 7

7: 7-orfs-openroad-route/.execstamp 7-orfs-openroad-route/.postconditions.stamp 7-orfs-openroad-route/outputs/.execstamp.flow-checkpoint.tar.gz 7-orfs-openroad-route/outputs/.stamp.flow-checkpoint.tar.gz

# debug

define 7_debug_rule
	cd 7-orfs-openroad-route && ./mflowgen-debug 2>&1 | tee mflowgen-debug.log
endef

debug-orfs-openroad-route: 
	$(call 7_debug_rule)

.PHONY: debug-7

debug-7: debug-orfs-openroad-route

# ------------------------------------------------------------------------
# orfs-openroad-finish
# ------------------------------------------------------------------------

# build dir

ifeq ("$(wildcard 8-orfs-openroad-finish/.prebuilt)","")
8-orfs-openroad-finish/.stamp: 2-orfs-docker-setup/.execstamp 2-orfs-docker-setup/.postconditions.stamp 2-orfs-docker-setup/outputs/.execstamp.orfs-docker-image.tar 2-orfs-docker-setup/outputs/.stamp.orfs-docker-image.tar
8-orfs-openroad-finish/.stamp: 7-orfs-openroad-route/.execstamp 7-orfs-openroad-route/.postconditions.stamp 7-orfs-openroad-route/outputs/.execstamp.flow-checkpoint.tar.gz 7-orfs-openroad-route/outputs/.stamp.flow-checkpoint.tar.gz
8-orfs-openroad-finish/.stamp:
	$(call cpdir-and-parameterize,8-orfs-openroad-finish,../nodes/orfs-openroad-finish,8-orfs-openroad-finish/.stamp)
endif

# collect inputs

ifeq ("$(wildcard 8-orfs-openroad-finish/.prebuilt)","")
8-orfs-openroad-finish/inputs/.stamp.flow-checkpoint.tar.gz: 8-orfs-openroad-finish/.stamp 7-orfs-openroad-route/outputs/.stamp.flow-checkpoint.tar.gz
	$(call symlink,8-orfs-openroad-finish/inputs,flow-checkpoint.tar.gz,../../7-orfs-openroad-route/outputs/flow-checkpoint.tar.gz,.stamp.flow-checkpoint.tar.gz)
endif

ifeq ("$(wildcard 8-orfs-openroad-finish/.prebuilt)","")
8-orfs-openroad-finish/inputs/.stamp.orfs-docker-image.tar: 8-orfs-openroad-finish/.stamp 2-orfs-docker-setup/outputs/.stamp.orfs-docker-image.tar
	$(call symlink,8-orfs-openroad-finish/inputs,orfs-docker-image.tar,../../2-orfs-docker-setup/outputs/orfs-docker-image.tar,.stamp.orfs-docker-image.tar)
endif

# execute

ifeq ("$(wildcard 8-orfs-openroad-finish/.prebuilt)","")

define 8_orfs_openroad_finish_commands_rule
	mkdir -p 8-orfs-openroad-finish/outputs && set -o pipefail && /home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-letters -c -t orfs-openroad-finish && chmod +x .mflowgen/8-orfs-openroad-finish/mflowgen-run && cp -f .mflowgen/8-orfs-openroad-finish/mflowgen-run 8-orfs-openroad-finish && if [[ -e .mflowgen/8-orfs-openroad-finish/mflowgen-debug ]]; then chmod +x .mflowgen/8-orfs-openroad-finish/mflowgen-debug && cp -f .mflowgen/8-orfs-openroad-finish/mflowgen-debug 8-orfs-openroad-finish; fi && if [[ -e .mflowgen/8-orfs-openroad-finish/mflowgen-check-preconditions.py ]]; then chmod +x .mflowgen/8-orfs-openroad-finish/mflowgen-check-preconditions.py && cp -f .mflowgen/8-orfs-openroad-finish/mflowgen-check-preconditions.py 8-orfs-openroad-finish; fi && if [[ -e .mflowgen/8-orfs-openroad-finish/mflowgen-check-postconditions.py ]]; then chmod +x .mflowgen/8-orfs-openroad-finish/mflowgen-check-postconditions.py && cp -f .mflowgen/8-orfs-openroad-finish/mflowgen-check-postconditions.py 8-orfs-openroad-finish; fi && cd 8-orfs-openroad-finish && if [[ -e mflowgen-check-preconditions.py ]]; then ./mflowgen-check-preconditions.py || exit 1; fi && ./mflowgen-run 2>&1 | tee mflowgen-run.log || exit 1 && cd .. && touch -c 8-orfs-openroad-finish/outputs/*
endef

8-orfs-openroad-finish/.execstamp: 8-orfs-openroad-finish/.stamp 8-orfs-openroad-finish/inputs/.stamp.flow-checkpoint.tar.gz 8-orfs-openroad-finish/inputs/.stamp.orfs-docker-image.tar
	$(call 8_orfs_openroad_finish_commands_rule)
	touch $@

8-orfs-openroad-finish/outputs/.execstamp.flow-checkpoint.tar.gz: 8-orfs-openroad-finish/.execstamp
	touch $@

endif

# collect outputs

8-orfs-openroad-finish/outputs/.stamp.flow-checkpoint.tar.gz: 8-orfs-openroad-finish/.execstamp 8-orfs-openroad-finish/outputs/.execstamp.flow-checkpoint.tar.gz
	$(call stamp,8-orfs-openroad-finish/outputs/.stamp.flow-checkpoint.tar.gz)

# post-conditions

define 8_orfs_openroad_finish_post_conditions_commands_rule
	cd 8-orfs-openroad-finish && if [[ -e mflowgen-check-postconditions.py ]]; then ./mflowgen-check-postconditions.py || exit 1; fi && cd ..
endef

8-orfs-openroad-finish/.postconditions.stamp: 8-orfs-openroad-finish/.execstamp 8-orfs-openroad-finish/outputs/.execstamp.flow-checkpoint.tar.gz 8-orfs-openroad-finish/outputs/.stamp.flow-checkpoint.tar.gz
	$(call 8_orfs_openroad_finish_post_conditions_commands_rule)
	touch $@

# alias

.PHONY: orfs-openroad-finish

orfs-openroad-finish: 8-orfs-openroad-finish/.execstamp 8-orfs-openroad-finish/.postconditions.stamp 8-orfs-openroad-finish/outputs/.execstamp.flow-checkpoint.tar.gz 8-orfs-openroad-finish/outputs/.stamp.flow-checkpoint.tar.gz

.PHONY: 8

8: 8-orfs-openroad-finish/.execstamp 8-orfs-openroad-finish/.postconditions.stamp 8-orfs-openroad-finish/outputs/.execstamp.flow-checkpoint.tar.gz 8-orfs-openroad-finish/outputs/.stamp.flow-checkpoint.tar.gz

# debug

define 8_debug_rule
	cd 8-orfs-openroad-finish && ./mflowgen-debug 2>&1 | tee mflowgen-debug.log
endef

debug-orfs-openroad-finish: 
	$(call 8_debug_rule)

.PHONY: debug-8

debug-8: debug-orfs-openroad-finish

# ------------------------------------------------------------------------
# Misc
# ------------------------------------------------------------------------

# Clean

.PHONY: clean-all

clean-all:
	@find . -maxdepth 1 ! -name Makefile ! -name .mflowgen\* ! -name \. ! -name \.\. -exec rm -rf {} +

.PHONY: clean-0

clean-0:
	rm -rf ./0-info

.PHONY: clean-info

clean-info: clean-0

.PHONY: clean-1

clean-1:
	rm -rf ./1-orfs-design

.PHONY: clean-orfs-design

clean-orfs-design: clean-1

.PHONY: clean-2

clean-2:
	rm -rf ./2-orfs-docker-setup

.PHONY: clean-orfs-docker-setup

clean-orfs-docker-setup: clean-2

.PHONY: clean-3

clean-3:
	rm -rf ./3-orfs-yosys-synthesis

.PHONY: clean-orfs-yosys-synthesis

clean-orfs-yosys-synthesis: clean-3

.PHONY: clean-4

clean-4:
	rm -rf ./4-orfs-openroad-floorplan

.PHONY: clean-orfs-openroad-floorplan

clean-orfs-openroad-floorplan: clean-4

.PHONY: clean-5

clean-5:
	rm -rf ./5-orfs-openroad-place

.PHONY: clean-orfs-openroad-place

clean-orfs-openroad-place: clean-5

.PHONY: clean-6

clean-6:
	rm -rf ./6-orfs-openroad-cts

.PHONY: clean-orfs-openroad-cts

clean-orfs-openroad-cts: clean-6

.PHONY: clean-7

clean-7:
	rm -rf ./7-orfs-openroad-route

.PHONY: clean-orfs-openroad-route

clean-orfs-openroad-route: clean-7

.PHONY: clean-8

clean-8:
	rm -rf ./8-orfs-openroad-finish

.PHONY: clean-orfs-openroad-finish

clean-orfs-openroad-finish: clean-8

# Diff

.PHONY: diff-0

diff-0:
	@echo && diff -r -u --minimal --exclude={configure.yml,.time_end,.time_start,mflowgen-run*,mflowgen-debug,.stamp,inputs,outputs} ../nodes/info 0-info | grep --color=always -e '^-.*' -e '$$' -e 'Only in ../nodes/info.*' | GREP_COLOR='01;32' grep --color=always -e '^+.*' -e '$$' -e 'Only in 0-info.*' && echo || true

.PHONY: diff-1

diff-1:
	@echo && diff -r -u --minimal --exclude={configure.yml,.time_end,.time_start,mflowgen-run*,mflowgen-debug,.stamp,inputs,outputs} ../designs/GcdUnit/orfs-design 1-orfs-design | grep --color=always -e '^-.*' -e '$$' -e 'Only in ../designs/GcdUnit/orfs-design.*' | GREP_COLOR='01;32' grep --color=always -e '^+.*' -e '$$' -e 'Only in 1-orfs-design.*' && echo || true

.PHONY: diff-2

diff-2:
	@echo && diff -r -u --minimal --exclude={configure.yml,.time_end,.time_start,mflowgen-run*,mflowgen-debug,.stamp,inputs,outputs} ../nodes/orfs-docker-setup 2-orfs-docker-setup | grep --color=always -e '^-.*' -e '$$' -e 'Only in ../nodes/orfs-docker-setup.*' | GREP_COLOR='01;32' grep --color=always -e '^+.*' -e '$$' -e 'Only in 2-orfs-docker-setup.*' && echo || true

.PHONY: diff-3

diff-3:
	@echo && diff -r -u --minimal --exclude={configure.yml,.time_end,.time_start,mflowgen-run*,mflowgen-debug,.stamp,inputs,outputs} ../nodes/orfs-yosys-synthesis 3-orfs-yosys-synthesis | grep --color=always -e '^-.*' -e '$$' -e 'Only in ../nodes/orfs-yosys-synthesis.*' | GREP_COLOR='01;32' grep --color=always -e '^+.*' -e '$$' -e 'Only in 3-orfs-yosys-synthesis.*' && echo || true

.PHONY: diff-4

diff-4:
	@echo && diff -r -u --minimal --exclude={configure.yml,.time_end,.time_start,mflowgen-run*,mflowgen-debug,.stamp,inputs,outputs} ../nodes/orfs-openroad-floorplan 4-orfs-openroad-floorplan | grep --color=always -e '^-.*' -e '$$' -e 'Only in ../nodes/orfs-openroad-floorplan.*' | GREP_COLOR='01;32' grep --color=always -e '^+.*' -e '$$' -e 'Only in 4-orfs-openroad-floorplan.*' && echo || true

.PHONY: diff-5

diff-5:
	@echo && diff -r -u --minimal --exclude={configure.yml,.time_end,.time_start,mflowgen-run*,mflowgen-debug,.stamp,inputs,outputs} ../nodes/orfs-openroad-place 5-orfs-openroad-place | grep --color=always -e '^-.*' -e '$$' -e 'Only in ../nodes/orfs-openroad-place.*' | GREP_COLOR='01;32' grep --color=always -e '^+.*' -e '$$' -e 'Only in 5-orfs-openroad-place.*' && echo || true

.PHONY: diff-6

diff-6:
	@echo && diff -r -u --minimal --exclude={configure.yml,.time_end,.time_start,mflowgen-run*,mflowgen-debug,.stamp,inputs,outputs} ../nodes/orfs-openroad-cts 6-orfs-openroad-cts | grep --color=always -e '^-.*' -e '$$' -e 'Only in ../nodes/orfs-openroad-cts.*' | GREP_COLOR='01;32' grep --color=always -e '^+.*' -e '$$' -e 'Only in 6-orfs-openroad-cts.*' && echo || true

.PHONY: diff-7

diff-7:
	@echo && diff -r -u --minimal --exclude={configure.yml,.time_end,.time_start,mflowgen-run*,mflowgen-debug,.stamp,inputs,outputs} ../nodes/orfs-openroad-route 7-orfs-openroad-route | grep --color=always -e '^-.*' -e '$$' -e 'Only in ../nodes/orfs-openroad-route.*' | GREP_COLOR='01;32' grep --color=always -e '^+.*' -e '$$' -e 'Only in 7-orfs-openroad-route.*' && echo || true

.PHONY: diff-8

diff-8:
	@echo && diff -r -u --minimal --exclude={configure.yml,.time_end,.time_start,mflowgen-run*,mflowgen-debug,.stamp,inputs,outputs} ../nodes/orfs-openroad-finish 8-orfs-openroad-finish | grep --color=always -e '^-.*' -e '$$' -e 'Only in ../nodes/orfs-openroad-finish.*' | GREP_COLOR='01;32' grep --color=always -e '^+.*' -e '$$' -e 'Only in 8-orfs-openroad-finish.*' && echo || true

# Info

.PHONY: info-0

info-0:
	@/home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-letters -c -t info && /home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-info -y .mflowgen/0-info/configure.yml

.PHONY: info-1

info-1:
	@/home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-letters -c -t orfs-design && /home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-info -y .mflowgen/1-orfs-design/configure.yml

.PHONY: info-2

info-2:
	@/home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-letters -c -t orfs-docker-setup && /home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-info -y .mflowgen/2-orfs-docker-setup/configure.yml

.PHONY: info-3

info-3:
	@/home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-letters -c -t orfs-yosys-synthesis && /home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-info -y .mflowgen/3-orfs-yosys-synthesis/configure.yml

.PHONY: info-4

info-4:
	@/home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-letters -c -t orfs-openroad-floorplan && /home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-info -y .mflowgen/4-orfs-openroad-floorplan/configure.yml

.PHONY: info-5

info-5:
	@/home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-letters -c -t orfs-openroad-place && /home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-info -y .mflowgen/5-orfs-openroad-place/configure.yml

.PHONY: info-6

info-6:
	@/home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-letters -c -t orfs-openroad-cts && /home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-info -y .mflowgen/6-orfs-openroad-cts/configure.yml

.PHONY: info-7

info-7:
	@/home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-letters -c -t orfs-openroad-route && /home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-info -y .mflowgen/7-orfs-openroad-route/configure.yml

.PHONY: info-8

info-8:
	@/home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-letters -c -t orfs-openroad-finish && /home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-info -y .mflowgen/8-orfs-openroad-finish/configure.yml

# Runtimes

.PHONY: runtimes

runtimes:
	@/home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-runtimes

# List

.PHONY: list

list:
	@echo && echo Generic Targets: && echo && printf " - %s\\n" "list      -- List all nodes" "status    -- Print build status for each node" "runtimes  -- Print runtimes for each node" "graph     -- Generate a PDF of the node dependency graph" "clean-all -- Remove all build directories" "clean-N   -- Clean target N" "info-N    -- Print configured info for node N" "diff-N    -- Diff target N" && echo && echo Targets: && echo && printf " - %s\\n" "  0 : info" "  1 : orfs-design" "  2 : orfs-docker-setup" "  3 : orfs-yosys-synthesis" "  4 : orfs-openroad-floorplan" "  5 : orfs-openroad-place" "  6 : orfs-openroad-cts" "  7 : orfs-openroad-route" "  8 : orfs-openroad-finish" && echo && echo Debug Targets: && echo && printf " - %s\\n" "debug-3  : debug-orfs-yosys-synthesis" "debug-4  : debug-orfs-openroad-floorplan" "debug-5  : debug-orfs-openroad-place" "debug-6  : debug-orfs-openroad-cts" "debug-7  : debug-orfs-openroad-route" "debug-8  : debug-orfs-openroad-finish" && echo

# Graph

.PHONY: graph

graph:
	dot -Tpdf .mflowgen/graph.dot > graph.pdf

# Status

.PHONY: status

status:
	@/home/sychae/cuda-vscode/mflowgen/mflowgen/scripts/mflowgen-status --backend make -s 0-info,1-orfs-design,2-orfs-docker-setup,3-orfs-yosys-synthesis,4-orfs-openroad-floorplan,5-orfs-openroad-place,6-orfs-openroad-cts,7-orfs-openroad-route,8-orfs-openroad-finish

