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
    property int selectedFixtureIndex: reviewFixtures.length > 0 ? 0 : -1
    property string queueStatusText: initialQueueStatus
    property string selectedMessageText: "Select a fixture to begin review."
    property string codexStreamText: ""
    property bool hasFixture: selectedFixtureIndex >= 0

    function currentFixture() {
        if (!hasFixture) {
            return null
        }
        return reviewFixtures[selectedFixtureIndex]
    }

    function selectFixture(index) {
        if (index < 0 || index >= reviewFixtures.length) {
            selectedFixtureIndex = -1
            selectedMessageText = "No fixture selected."
            return
        }
        selectedFixtureIndex = index
        var fixture = currentFixture()
        selectedMessageText = fixture.fixture_id
    }

    Component.onCompleted: {
        if (reviewFixtures.length > 0) {
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
                            root.queueStatusText = root.reviewFixtures.length > 0
                                ? root.reviewFixtures.length + " fixture" + (root.reviewFixtures.length === 1 ? "" : "s") + " loaded"
                                : "No fixtures loaded"
                            if (root.reviewFixtures.length > 0) {
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
                    tone: root.reviewFixtures.length > 0 ? "ready" : "warning"
                }

                ListView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    model: root.reviewFixtures
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
                                text: modelData.artifact_display_path || modelData.source_display_path
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

                    Components.StatusBadge {
                        label: startupDiagnostics.length > 0 ? "Diagnostics" : "Ready"
                        tone: startupDiagnostics.length > 0 ? "warning" : "ready"
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
                    Layout.minimumHeight: 240
                    radius: 4
                    border.color: "#c9c8be"
                    color: "#ffffff"

                    Label {
                        anchors.centerIn: parent
                        width: Math.min(parent.width - 48, 520)
                        text: root.selectedMessageText
                        horizontalAlignment: Text.AlignHCenter
                        wrapMode: Text.WordWrap
                        color: "#565a51"
                    }
                }

                Components.MarkdownPanel {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.minimumHeight: 240
                    markdownText: root.hasFixture
                        ? "# " + root.currentFixture().fixture_id + "\n\nSource: `" + root.currentFixture().source_display_path + "`\n\nExpected: " + (root.currentFixture().expected_output || "not declared")
                            + (root.currentFixture().artifact_display_path ? "\n\nArtifact: `" + root.currentFixture().artifact_display_path + "`" : "")
                        : "No fixture context loaded."
                }

                Components.ArtifactPanel {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 180
                    artifactCount: 0
                }

                Components.NotesPanel {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 180
                    noteText: ""
                }

                Components.CodexPanel {
                    Layout.columnSpan: 2
                    Layout.fillWidth: true
                    Layout.preferredHeight: 220
                    streamText: root.codexStreamText
                    onSendRequested: function(prompt) {
                        root.codexStreamText = root.hasFixture
                            ? "Request queued."
                            : "No fixture selected."
                    }
                }
            }
        }
    }
}
