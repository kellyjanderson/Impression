import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "components" as Components

ApplicationWindow {
    id: root
    width: 1180
    height: 760
    minimumWidth: 880
    minimumHeight: 560
    visible: true
    title: "Reference Review Workbench"
    color: "#f7f7f2"
    property var reviewFixtures: fixtureItems
    property bool showApproved: false
    property var filteredReviewFixtures: reviewFixtures.filter(function(fixture) {
        return root.showApproved || fixture.status !== "approved"
    })
    property int selectedFixtureIndex: filteredReviewFixtures.length > 0 ? 0 : -1
    property string queueStatusText: initialQueueStatus
    property string selectedMessageText: "Select a fixture to begin review."
    property bool hasFixture: selectedFixtureIndex >= 0

    function currentFixture() {
        if (!hasFixture) {
            return null
        }
        return filteredReviewFixtures[selectedFixtureIndex]
    }

    function selectFixture(index) {
        if (index < 0 || index >= filteredReviewFixtures.length) {
            selectedFixtureIndex = -1
            selectedMessageText = "No fixture selected."
            return
        }
        selectedFixtureIndex = index
        var fixture = currentFixture()
        selectedMessageText = fixture.fixture_id
    }

    function reviewStatusLabel() {
        if (!hasFixture) {
            return "UNREVIEWED"
        }
        var status = currentFixture().status || "unreviewed"
        if (status === "approved") {
            return "APPROVED"
        }
        if (status === "declined") {
            return "DECLINED"
        }
        return "UNREVIEWED"
    }

    function reviewStatusColor() {
        if (!hasFixture) {
            return "#5f6368"
        }
        var status = currentFixture().status || "unreviewed"
        if (status === "approved") {
            return "#1f7a4d"
        }
        if (status === "declined") {
            return "#b42318"
        }
        return "#5f6368"
    }

    Component.onCompleted: {
        if (filteredReviewFixtures.length > 0) {
            selectFixture(0)
        } else {
            selectedMessageText = "Load a fixture file or database to begin review."
        }
    }

    SplitView {
        anchors.fill: parent
        orientation: Qt.Horizontal

        Frame {
            SplitView.preferredWidth: 280
            SplitView.minimumWidth: 220
            SplitView.maximumWidth: 360
            padding: 12

            ColumnLayout {
                anchors.fill: parent
                spacing: 10

                RowLayout {
                    Layout.fillWidth: true

                    Text {
                        Layout.fillWidth: true
                        text: "Queue"
                        font.pixelSize: 18
                        font.bold: true
                        color: "#242622"
                        elide: Text.ElideRight
                    }

                    Button {
                        objectName: "refreshQueueButton"
                        text: "Refresh"
                        onClicked: {
                            root.queueStatusText = root.filteredReviewFixtures.length > 0
                                ? root.filteredReviewFixtures.length + " fixture" + (root.filteredReviewFixtures.length === 1 ? "" : "s") + " shown"
                                : "No fixtures loaded"
                            if (root.filteredReviewFixtures.length > 0) {
                                root.selectFixture(Math.max(root.selectedFixtureIndex, 0))
                            } else {
                                root.selectFixture(-1)
                            }
                        }
                    }
                }

                Components.StatusBadge {
                    Layout.fillWidth: true
                    label: root.queueStatusText
                    tone: root.filteredReviewFixtures.length > 0 ? "ready" : "warning"
                }

                CheckBox {
                    objectName: "showApprovedCheckBox"
                    text: "show approved"
                    checked: false
                    onToggled: {
                        root.showApproved = checked
                        root.selectFixture(root.filteredReviewFixtures.length > 0 ? 0 : -1)
                    }
                }

                ListView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    model: root.filteredReviewFixtures
                    delegate: ItemDelegate {
                        width: ListView.view.width
                        height: 44
                        text: modelData.fixture_id
                        highlighted: index === root.selectedFixtureIndex
                        onClicked: root.selectFixture(index)

                        contentItem: Column {
                            spacing: 1
                            Text {
                                width: parent.width
                                text: modelData.fixture_id
                                color: "#242622"
                                font.pixelSize: 13
                                elide: Text.ElideMiddle
                            }
                            Text {
                                width: parent.width
                                text: (modelData.status || "unreviewed") + " - " + (modelData.artifact_kind_label || modelData.source_display_path)
                                color: "#565a51"
                                font.pixelSize: 11
                                elide: Text.ElideMiddle
                            }
                        }
                    }
                }
            }
        }

        Frame {
            SplitView.fillWidth: true
            padding: 16

            GridLayout {
                anchors.fill: parent
                columns: 2
                columnSpacing: 12
                rowSpacing: 12

                RowLayout {
                    Layout.columnSpan: 2
                    Layout.fillWidth: true
                    spacing: 8

                    Text {
                        Layout.fillWidth: true
                        text: "Selected Fixture"
                        font.pixelSize: 18
                        font.bold: true
                        color: "#242622"
                        elide: Text.ElideRight
                    }

                    Button {
                        objectName: "approveFixtureButton"
                        text: startupDiagnostics.length > 0 ? "Diagnostics" : "Approve"
                        enabled: root.hasFixture && startupDiagnostics.length === 0
                    }

                    Button {
                        objectName: "declineFixtureButton"
                        text: "Decline"
                        enabled: root.hasFixture
                    }

                    Button {
                        objectName: "previousFixtureButton"
                        text: "Previous"
                        enabled: root.hasFixture
                        onClicked: root.selectFixture(Math.max(0, root.selectedFixtureIndex - 1))
                    }

                    Button {
                        objectName: "nextFixtureButton"
                        text: "Next"
                        enabled: root.hasFixture
                        onClicked: root.selectFixture(Math.min(root.reviewFixtures.length - 1, root.selectedFixtureIndex + 1))
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.preferredWidth: 360
                    Layout.minimumWidth: 300
                    Layout.minimumHeight: 240
                    radius: 4
                    border.color: "#c9c8be"
                    color: "#ffffff"

                    Image {
                        id: selectedArtifactPreview
                        anchors.fill: parent
                        anchors.margins: 12
                        source: root.hasFixture ? root.currentFixture().artifact_preview_url : ""
                        fillMode: Image.PreserveAspectFit
                        asynchronous: true
                        cache: false
                        visible: source !== "" && status === Image.Ready
                    }

                    Label {
                        anchors.centerIn: parent
                        width: Math.min(parent.width - 48, 520)
                        text: root.hasFixture && root.currentFixture().artifact_preview_url === ""
                            ? (root.currentFixture().preview_empty_message || "No STL or .impress preview is available.")
                            : root.selectedMessageText
                        horizontalAlignment: Text.AlignHCenter
                        wrapMode: Text.WordWrap
                        color: "#565a51"
                        visible: !selectedArtifactPreview.visible
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 30
                    radius: 4
                    color: root.reviewStatusColor()
                    border.color: "#ffffff"
                    border.width: 1

                    Text {
                        anchors.centerIn: parent
                        text: root.reviewStatusLabel()
                        color: "#ffffff"
                        font.pixelSize: 12
                        font.bold: true
                    }
                }

                Components.MarkdownPanel {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.minimumHeight: 240
                    markdownText: root.hasFixture
                        ? "# " + root.currentFixture().fixture_id
                            + "\n\nReview: " + (root.currentFixture().status || "unreviewed")
                            + "\n\nPurpose: " + (root.currentFixture().purpose || "not provided")
                            + "\n\nMethodology: " + (root.currentFixture().methodology || "not provided")
                            + "\n\nRendered result: " + (root.currentFixture().render_description || "not provided")
                            + "\n\nSource: `" + root.currentFixture().source_display_path + "`\n\nExpected: " + (root.currentFixture().expected_output || "not declared")
                            + (root.currentFixture().artifact_display_path ? "\n\nArtifact: `" + root.currentFixture().artifact_display_path + "`" : "")
                        : "No fixture context loaded."
                }

                Components.ArtifactPanel {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 180
                    Layout.preferredWidth: 360
                    Layout.minimumWidth: 300
                    artifacts: root.hasFixture && root.currentFixture().artifact_display_path
                        ? [root.currentFixture()]
                        : []
                }

                Components.NotesPanel {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 180
                    noteText: ""
                }
            }
        }
    }
}
